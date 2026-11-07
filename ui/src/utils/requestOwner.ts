// Response ownership for polling hooks (docs/STATE_MACHINES.md M18, C07).
//
// Every request takes a ticket. A response (and its state updates, cursor writes, status, in-flight release) is
// applied only if its ticket still belongs to the current identity (the job/ref/... the hook is showing), no
// mutation was made since it was issued (a GET issued before a DELETE may carry pre-DELETE data), and nothing newer
// has been applied already (manual refreshes can resolve out of order). The owner also remembers WHICH identity is
// current (the key passed to reset), so a callback created for an older identity (a stale closure, a queued timer)
// can tell that it must not start a request under the new identity's tickets.
export type Ticket = { identity: number; seq: number; epoch: number };

export class RequestOwner {
  private identity = 0;
  private seq = 0;
  private epoch = 0;
  private applied = 0;
  private key: unknown = undefined;

  // The thing being shown changed (to key): everything issued before is stale.
  reset(key?: unknown) {
    this.identity += 1;
    this.applied = this.seq;
    this.key = key;
  }

  // Is key the identity being shown now?
  isCurrent(key: unknown): boolean {
    return Object.is(key, this.key);
  }

  begin(): Ticket {
    this.seq += 1;
    return { identity: this.identity, seq: this.seq, epoch: this.epoch };
  }

  // A state change was made (or is being made) by this client: every request issued before now is stale.
  mutate(): Ticket {
    this.epoch += 1;
    this.applied = this.seq;
    return this.begin();
  }

  // Same identity and no mutation since the ticket was issued.
  valid(t: Ticket): boolean {
    return t.identity === this.identity && t.epoch === this.epoch;
  }

  // Same identity (the request's in-flight slot/finally still belongs to this hook instance's current target).
  owns(t: Ticket): boolean {
    return t.identity === this.identity;
  }

  // Claim the right to apply a response: valid, and newer than anything applied. Consumes the ticket.
  accept(t: Ticket): boolean {
    if (!this.valid(t) || t.seq <= this.applied) return false;
    this.applied = t.seq;
    return true;
  }
}
