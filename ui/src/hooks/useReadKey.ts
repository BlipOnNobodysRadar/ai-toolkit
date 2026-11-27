'use client';

import { useSyncExternalStore } from 'react';
import { currentReadKey, subscribeReadKey } from '@/utils/readKey';

export default function useReadKey() {
  return useSyncExternalStore(subscribeReadKey, currentReadKey, () => '');
}
