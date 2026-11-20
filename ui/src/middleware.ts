// middleware.ts (at the root of your project)
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { isReadProtectionDisabled, isValidReadKey, READ_KEY_PARAM } from '@/server/readKey';

// Album art and dataset media are fetched by plain <img>/<video> tags, which cannot send the bearer token. These are
// read-only exemptions only: a mutating route must never become public merely because it shares a prefix. Reads on
// them pass with a live read key instead of the token (upstream 73c84b09).
const publicReadPrefixes = ['/api/img/', '/api/files/', '/api/audio/art/'];
const PUBLIC_READ_METHODS = new Set(['GET', 'HEAD']);

const isPublicRead = (request: NextRequest) =>
  PUBLIC_READ_METHODS.has(request.method.toUpperCase()) &&
  publicReadPrefixes.some(route => request.nextUrl.pathname.startsWith(route));

export async function middleware(request: NextRequest) {
  // check env var for AI_TOOLKIT_AUTH, if not set, approve all requests
  // if it is set make sure bearer token matches
  const tokenToUse = process.env.AI_TOOLKIT_AUTH || null;
  if (!tokenToUse) {
    return NextResponse.next();
  }

  // Get the token from the headers
  const token = request.headers.get('Authorization')?.split(' ')[1];

  // public read routes pass with a live read key; otherwise the token check below applies
  if (isPublicRead(request)) {
    if (isReadProtectionDisabled() || (await isValidReadKey(request.nextUrl.searchParams.get(READ_KEY_PARAM)))) {
      return NextResponse.next();
    }
  }

  // Check if the route should be protected
  // This will apply to all API routes that start with /api/
  if (request.nextUrl.pathname.startsWith('/api/')) {
    if (!token || token !== tokenToUse) {
      // Return a JSON response with 401 Unauthorized
      return new NextResponse(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // For authorized users, continue
    return NextResponse.next();
  }

  // For non-API routes, just continue
  return NextResponse.next();
}

// Configure which paths this middleware will run on
export const config = {
  matcher: [
    // Apply to all API routes
    '/api/:path*',
  ],
};
