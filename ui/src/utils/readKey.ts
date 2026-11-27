// Read key for /api/img, /api/files and /api/audio/art URLs when a password is set.
// Empty otherwise. AuthWrapper sets it from /api/auth.
let readKey = '';
const listeners = new Set<() => void>();

export const currentReadKey = () => readKey;
export const subscribeReadKey = (notify: () => void) => {
  listeners.add(notify);
  return () => { listeners.delete(notify); };
};

export const setReadKey = (key: string) => {
  if (key === readKey) return;
  readKey = key;
  listeners.forEach(notify => notify());
};

// for URLs built without encodeFilePathForUrl, which appends the key itself
export const withReadKey = (url: string) => {
  if (!readKey) return url;
  return `${url}${url.includes('?') ? '&' : '?'}k=${readKey}`;
};
