import test from 'node:test';
import assert from 'node:assert/strict';
import {nextPair} from '../src/app/distill/review.ts';

test('blind schedule covers only unseen within-prompt pairs, chain first', () => {
  const rows=['A','B','C','D','E'].map((id,i)=>({id,group:i<3?0:1,seed:40+i}));
  const review={votes:[],skipped:[],excluded:[]};
  let pair=nextPair(rows,0,review);
  assert.deepEqual(pair.toSorted(),['A','B']);
  review.skipped.push(pair);
  pair=nextPair(rows,0,review);
  assert.deepEqual(pair.toSorted(),['B','C']);
  review.votes.push([...pair,'tie']);
  pair=nextPair(rows,0,review);
  assert.deepEqual(pair.toSorted(),['A','C']);
  review.votes.push([...pair,'b']);
  assert.equal(nextPair(rows,0,review),null);
  assert.deepEqual(nextPair(rows,1,review).toSorted(),['D','E']);
});
