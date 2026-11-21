'use client'

import { Suspense, useSyncExternalStore, type ReactNode } from 'react'

const subscribe = () => () => {}

/**
 * SWR's `suspense` mode cannot fetch during SSR without fallback data, so the
 * children only render in the browser. The server and hydration pass render
 * the fallback instead.
 */
export function ClientSuspense({
  children,
  fallback,
}: {
  children: ReactNode
  fallback: ReactNode
}) {
  const isClient = useSyncExternalStore(
    subscribe,
    () => true,
    () => false
  )

  if (!isClient) return fallback

  return <Suspense fallback={fallback}>{children}</Suspense>
}
