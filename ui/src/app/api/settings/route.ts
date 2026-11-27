import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { defaultTrainFolder, defaultDatasetsFolder, defaultModelsFolder } from '@/paths';
import { flushCache } from '@/server/settings';

const folderKeys = ['TRAINING_FOLDER', 'DATASETS_FOLDER', 'MODELS_PATH'] as const;

export async function GET() {
  try {
    const [folders, tokenCount] = await Promise.all([
      prisma.settings.findMany({ where: { key: { in: [...folderKeys] } } }),
      prisma.settings.count({ where: { key: 'HF_TOKEN', value: { not: '' } } }),
    ]);
    const values = Object.fromEntries(folders.map(({ key, value }) => [key, value]));
    return NextResponse.json({
      TRAINING_FOLDER: values.TRAINING_FOLDER || defaultTrainFolder,
      DATASETS_FOLDER: values.DATASETS_FOLDER || defaultDatasetsFolder,
      MODELS_PATH: process.env.MODELS_PATH?.trim() ? process.env.MODELS_PATH : values.MODELS_PATH || defaultModelsFolder,
      HAS_HF_TOKEN: tokenCount > 0,
    }, { headers: { 'Cache-Control': 'no-store' } });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const updates = [];
    for (const key of [...folderKeys, 'HF_TOKEN']) {
      if (!Object.hasOwn(body, key)) continue; // Omission preserves the stored value. An explicit empty token clears it.
      const value = body[key];
      if (typeof value !== 'string') {
        return NextResponse.json({ error: 'Invalid setting value' }, { status: 400 });
      }
      updates.push(prisma.settings.upsert({ where: { key }, update: { value }, create: { key, value } }));
    }
    await prisma.$transaction(updates);
    flushCache();
    return NextResponse.json({ success: true });
  } catch {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
