export type Vote = [string, string, 'a' | 'b' | 'tie'];
export type Row = {id: string; group: number; seed: number};
export type Review = { votes: Vote[]; excluded: string[]; skipped: string[][] };

// First build a chain so extremes are connected; then schedule wider gaps without
// materializing a quadratic list. Stable side shuffling keeps seed order blind.
export function nextPair(rows: Row[], group: number, review: Review): [string, string] | null {
  const ids = rows.filter(r => r.group === group).map(r => r.id);
  const seen = new Set([...review.votes, ...review.skipped].map(([a,b]) => [a,b].sort().join('/')));
  for (let gap = 1; gap < ids.length; gap++) {
    for (let i = 0; i + gap < ids.length; i++) {
      const a = ids[i], b = ids[i+gap];
      if (!seen.has([a,b].sort().join('/'))) {
        // Pseudorandom orientation only; no seed/model labels in the UI.
        let hash = 2166136261;
        for (const c of a + b) hash = Math.imul(hash ^ c.charCodeAt(0), 16777619);
        return ((hash >>> 12) & 1) ? [a,b] : [b,a];
      }
    }
  }
  return null;
}
