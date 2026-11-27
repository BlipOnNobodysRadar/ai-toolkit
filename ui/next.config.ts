import type { NextConfig } from 'next';
import { cpSync, readFileSync } from 'fs';
import { dirname, join } from 'path';
import { PHASE_DEVELOPMENT_SERVER, PHASE_PRODUCTION_BUILD } from 'next/constants';
import { randomBytes } from 'crypto';

process.env.NEXT_TELEMETRY_DISABLED ??= '1';

// per-launch secret behind the read keys; the front server (cron/fileServer.ts) sets it first when it launches Next
if (process.env.AI_TOOLKIT_AUTH && !process.env.AI_TOOLKIT_READ_SECRET) {
  process.env.AI_TOOLKIT_READ_SECRET = randomBytes(32).toString('hex');
}

const versionFile = readFileSync(join(__dirname, '..', 'version.py'), 'utf8');
const versionMatch = versionFile.match(/VERSION\s*=\s*["']([^"']+)["']/);
const appVersion = versionMatch ? versionMatch[1] : 'unknown';

const nextConfig: NextConfig = {
  // Media requests can contain private filenames and a short-lived read key.
  logging: { incomingRequests: false },
  env: {
    NEXT_PUBLIC_APP_VERSION: appVersion,
  },
  serverExternalPackages: ['macstats', 'osx-temperature-sensor', 'macos-temperature-sensor'],
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push('osx-temperature-sensor', 'macos-temperature-sensor', 'macstats');
    }
    return config;
  },
  devIndicators: false,
  experimental: {
    serverActions: {
      bodySizeLimit: '100gb',
    },
    middlewareClientMaxBodySize: '100gb',
  },
};

export default function config(phase: string): NextConfig {
  if (phase === PHASE_DEVELOPMENT_SERVER || phase === PHASE_PRODUCTION_BUILD) {
    const monaco = dirname(require.resolve('monaco-editor/package.json'));
    cpSync(join(monaco, 'min', 'vs'), join(__dirname, 'public', 'monaco', 'vs'), { recursive: true });
  }
  return nextConfig;
}
