//! Scaling benchmarks: how throughput changes with the number of cores, for
//! [`ConcurrentPriorityQueue`] compared with a `Mutex<BinaryHeap>` (strict priority order) and an
//! unbounded `ConcurrentQueue` (lock-free, but no priority order at all, so it is a throughput
//! reference rather than a drop-in alternative).
//!
//! The benchmark uses one worker thread per available core ([`thread::available_parallelism`],
//! which respects the process's CPU affinity). To measure scaling, run it several times with
//! different core counts, e.g. `taskset -c 0,2,4,6 cargo bench ...` for 4 cores (on machines with
//! SMT, pick one CPU per physical core first). Running fewer threads than cores, or more, would
//! measure something else: idle cores, or threads being preempted while they hold a lock.
//!
//! Every benchmark iteration does a fixed amount of work, split across the threads, so the time per
//! iteration directly shows how well an implementation scales: perfect scaling halves it when the
//! core count doubles. Benchmark ids are
//! `turbo_concurrent_priority_queue_scaling/<workload>/<mix>/<impl>/<threads>`, where `<threads>`
//! is the number of available cores.
//!
//! Workloads:
//! - `mixed1`, `mixed2`, `mixed128`, `mixed1024`: like a task scheduler, every thread pops an item
//!   and then pushes the item's children, until all work is done. The children of an item only
//!   depend on its value (see [`Fan`]), so the total work is the same for every run and every
//!   implementation (within 0.4% of [`TOTAL_OPS`] pops). `mixedN` items have `N` children, but only
//!   one in `N` items has any (`mixed1`: every item has one child; `mixed1024`: one in 1024 items
//!   has 1024 children). The queue is prefilled with the root items (not timed; they all land in
//!   the prefilling thread's home shard of the sharded queue, like in `drain`). A thread that finds
//!   the queue empty waits for other threads' children until all threads are idle;
//! - `drain`: the queue is prefilled with [`TOTAL_OPS`] items (not timed); all threads pop until it
//!   is empty, each exactly its share;
//! - `fill`: all threads only push, [`TOTAL_OPS`] items together; the queue is drained afterwards
//!   (not timed).
//!
//! Priority mixes:
//! - `single`: every item has priority 0;
//! - `skewed`: like a build: most items have the lowest priority, a few have one of three higher
//!   ones. The priority is computed from the low bits of the item value (cheap, see
//!   [`Mix::priority`]);
//! - `uniform128`: the items are spread evenly over 128 priorities (the low 7 bits of the item
//!   value).
//!
//! The worker threads are spawned once per Criterion sample; each iteration releases them with a
//! start barrier and waits for them with an end barrier. Only the time between the two (including
//! the barrier hand-off) is measured; prefilling, draining and checking happen outside of it. Every
//! iteration checks its item counts, so a broken queue makes the benchmark panic instead of
//! reporting a meaningless number.

use std::{
    collections::BinaryHeap,
    hint::black_box,
    sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use concurrent_queue::ConcurrentQueue;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parking_lot::Mutex;
use turbo_concurrent_priority_queue::ConcurrentPriorityQueue;

/// Items per iteration of every workload (all threads together): `fill` pushes and `drain` pops
/// exactly this many, `mixed*` pops within 0.4% of it.
const TOTAL_OPS: usize = 1 << 19;
/// How many generations of items with children a `mixed*` root item has (see [`Fan`]).
const FAN_STEPS: u64 = 15;

/// The operations the benchmarks need, implemented by every compared queue.
trait BenchQueue: Sync {
    const NAME: &'static str;
    fn new() -> Self;
    fn push(&self, priority: u32, item: u64);
    fn pop(&self) -> Option<u64>;
}

impl BenchQueue for ConcurrentPriorityQueue<u32, u64> {
    const NAME: &'static str = "sharded";

    fn new() -> Self {
        ConcurrentPriorityQueue::new()
    }

    fn push(&self, priority: u32, item: u64) {
        ConcurrentPriorityQueue::push(self, priority, item);
    }

    fn pop(&self) -> Option<u64> {
        ConcurrentPriorityQueue::pop(self).map(|(_, item)| item)
    }
}

struct MutexHeap(Mutex<BinaryHeap<(u32, u64)>>);

impl BenchQueue for MutexHeap {
    const NAME: &'static str = "mutex_heap";

    fn new() -> Self {
        MutexHeap(Mutex::new(BinaryHeap::new()))
    }

    fn push(&self, priority: u32, item: u64) {
        self.0.lock().push((priority, item));
    }

    fn pop(&self) -> Option<u64> {
        self.0.lock().pop().map(|(_, item)| item)
    }
}

struct Unordered(ConcurrentQueue<(u32, u64)>);

impl BenchQueue for Unordered {
    const NAME: &'static str = "concurrent_queue";

    fn new() -> Self {
        Unordered(ConcurrentQueue::unbounded())
    }

    fn push(&self, priority: u32, item: u64) {
        self.0
            .push((priority, item))
            .expect("an unbounded queue that is never closed accepts every push");
    }

    fn pop(&self) -> Option<u64> {
        self.0.pop().ok().map(|(_, item)| item)
    }
}

/// How `mixed*` items fan out. An item value is `steps << 32 | low`. An item with `steps > 0` and
/// a `low` that [`Fan::children`] accepts has that many children, with values
/// `(steps - 1) << 32 | (low + i)` for `i` in `1..=children`; exactly one of them has children
/// again (`low + children`, or `low + 1` for `mixed1`). So every root item starts a chain of
/// [`FAN_STEPS`] fan-outs, and the work only depends on the root items.
#[derive(Clone, Copy)]
enum Fan {
    /// Every item has 1 child.
    One,
    /// Odd items have 2 children.
    Two,
    /// Every 128th item has 128 children.
    OneTwentyEight,
    /// Every 1024th item has 1024 children.
    TenTwentyFour,
}

impl Fan {
    const ALL: [Fan; 4] = [Fan::One, Fan::Two, Fan::OneTwentyEight, Fan::TenTwentyFour];

    fn name(self) -> &'static str {
        match self {
            Fan::One => "mixed1",
            Fan::Two => "mixed2",
            Fan::OneTwentyEight => "mixed128",
            Fan::TenTwentyFour => "mixed1024",
        }
    }

    /// How many children an item with low value bits `low` (and `steps > 0`) has.
    fn children(self, low: u32) -> u32 {
        match self {
            Fan::One => 1,
            Fan::Two => {
                if low & 1 == 1 {
                    2
                } else {
                    0
                }
            }
            Fan::OneTwentyEight => {
                if low & (128 - 1) == 0 {
                    128
                } else {
                    0
                }
            }
            Fan::TenTwentyFour => {
                if low & (1024 - 1) == 0 {
                    1024
                } else {
                    0
                }
            }
        }
    }

    /// The number of root items, so that the total number of pops,
    /// `roots * (1 + children * FAN_STEPS)`, is at most and close to [`TOTAL_OPS`] (within 0.4%).
    fn roots(self) -> u32 {
        let chain = 1 + u64::from(self.children(self.root_low(0))) * FAN_STEPS;
        (TOTAL_OPS as u64 / chain).max(1) as u32
    }

    /// The low value bits of the `index`-th root item (one that has children).
    fn root_low(self, index: u32) -> u32 {
        match self {
            Fan::One => index,
            Fan::Two => 2 * index + 1,
            Fan::OneTwentyEight => 128 * index,
            Fan::TenTwentyFour => 1024 * index,
        }
    }

    /// The total number of pops of one iteration: every root item and all its descendants.
    fn total_pops(self) -> usize {
        self.roots() as usize * (1 + self.children(self.root_low(0)) as usize * FAN_STEPS as usize)
    }
}

#[derive(Clone, Copy)]
enum Workload {
    Fan(Fan),
    Drain,
    Fill,
}

impl Workload {
    fn all() -> impl Iterator<Item = Workload> {
        Fan::ALL
            .into_iter()
            .map(Workload::Fan)
            .chain([Workload::Drain, Workload::Fill])
    }

    fn name(self) -> &'static str {
        match self {
            Workload::Fan(fan) => fan.name(),
            Workload::Drain => "drain",
            Workload::Fill => "fill",
        }
    }

    /// Pops (`mixed*`, `drain`) or pushes (`fill`) of one iteration, all threads together.
    fn ops(self) -> usize {
        match self {
            Workload::Fan(fan) => fan.total_pops(),
            Workload::Drain | Workload::Fill => TOTAL_OPS,
        }
    }
}

#[derive(Clone, Copy)]
enum Mix {
    Single,
    Skewed,
    Uniform128,
}

impl Mix {
    const ALL: [Mix; 3] = [Mix::Single, Mix::Skewed, Mix::Uniform128];

    fn name(self) -> &'static str {
        match self {
            Mix::Single => "single",
            Mix::Skewed => "skewed",
            Mix::Uniform128 => "uniform128",
        }
    }

    /// The priority of an item, from the low bits of its value: for `skewed`, 12 in 16 items get
    /// priority 0, 2 get 1, 1 gets 2 and 1 gets 3; for `uniform128`, the low 7 bits, so the items
    /// are spread evenly over 128 priorities.
    fn priority(self, item: u64) -> u32 {
        match self {
            Mix::Single => 0,
            Mix::Skewed => match item & 15 {
                0..=11 => 0,
                12 | 13 => 1,
                14 => 2,
                _ => 3,
            },
            Mix::Uniform128 => (item & 127) as u32,
        }
    }
}

/// A unique item value for the `index`-th item pushed by thread `thread` (`drain`, `fill`).
fn item(thread: usize, index: usize) -> u64 {
    (thread as u64) << 32 | index as u64
}

/// Thread index used for items pushed outside of the timed region (prefill).
const SETUP_THREAD: usize = u32::MAX as usize;

fn push_item<Q: BenchQueue>(queue: &Q, mix: Mix, item: u64) {
    queue.push(mix.priority(item), item);
}

/// Pops an item, or, when the queue is empty, waits until another thread pushes one. Returns
/// `None` once all `threads` threads are waiting (all work is done). `idle` counts the waiting
/// threads; it is only touched when a pop finds the queue empty.
fn pop_or_wait<Q: BenchQueue>(queue: &Q, idle: &AtomicUsize, threads: usize) -> Option<u64> {
    if let Some(item) = queue.pop() {
        return Some(item);
    }
    idle.fetch_add(1, Ordering::SeqCst);
    loop {
        if let Some(item) = queue.pop() {
            idle.fetch_sub(1, Ordering::SeqCst);
            return Some(item);
        }
        // A thread that holds a popped item hasn't counted itself as idle, so this can't be
        // reached while one still has children to push... except in the window between another
        // waiting thread's successful pop and its `fetch_sub`. Then this thread stops early, and
        // the remaining work is done by the threads that hold items, which never stop before
        // they find the queue empty themselves.
        if idle.load(Ordering::SeqCst) == threads {
            return None;
        }
        std::hint::spin_loop();
    }
}

/// The work one thread does in one iteration. Returns how many items it popped.
fn work<Q: BenchQueue>(
    queue: &Q,
    workload: Workload,
    mix: Mix,
    thread: usize,
    threads: usize,
    idle: &AtomicUsize,
) -> usize {
    // The first `ops % threads` threads take one item more, so the shares add up to `ops` for any
    // core count.
    let share = workload.ops() / threads + usize::from(thread < workload.ops() % threads);
    match workload {
        Workload::Fan(fan) => {
            let mut popped = 0;
            while let Some(item) = pop_or_wait(queue, idle, threads) {
                popped += 1;
                let steps = item >> 32;
                let low = item as u32;
                if steps > 0 {
                    let child_steps = (steps - 1) << 32;
                    for i in 1..=fan.children(low) {
                        push_item(queue, mix, child_steps | u64::from(low + i));
                    }
                }
            }
            popped
        }
        Workload::Drain => {
            for _ in 0..share {
                black_box(
                    queue
                        .pop()
                        .expect("drain: every thread pops exactly its share"),
                );
            }
            share
        }
        Workload::Fill => {
            for index in 0..share {
                push_item(queue, mix, item(thread, index));
            }
            0
        }
    }
}

fn prefill<Q: BenchQueue>(queue: &Q, workload: Workload, mix: Mix) {
    match workload {
        Workload::Fan(fan) => {
            for index in 0..fan.roots() {
                push_item(queue, mix, FAN_STEPS << 32 | u64::from(fan.root_low(index)));
            }
        }
        Workload::Drain => {
            for index in 0..TOTAL_OPS {
                push_item(queue, mix, item(SETUP_THREAD, index));
            }
        }
        Workload::Fill => {}
    }
}

fn drain_count<Q: BenchQueue>(queue: &Q) -> usize {
    std::iter::from_fn(|| queue.pop()).count()
}

/// Runs `iters` iterations with `threads` worker threads and returns the measured time.
fn run<Q: BenchQueue>(workload: Workload, mix: Mix, threads: usize, iters: u64) -> Duration {
    let queue = Q::new();
    let start = Barrier::new(threads + 1);
    let end = Barrier::new(threads + 1);
    let idle = AtomicUsize::new(0);
    let popped = AtomicUsize::new(0);
    thread::scope(|scope| {
        for thread in 0..threads {
            let (queue, start, end, idle, popped) = (&queue, &start, &end, &idle, &popped);
            scope.spawn(move || {
                for _ in 0..iters {
                    start.wait();
                    let count = work(queue, workload, mix, thread, threads, idle);
                    popped.fetch_add(count, Ordering::Relaxed);
                    end.wait();
                }
            });
        }
        let mut elapsed = Duration::ZERO;
        for _ in 0..iters {
            // Workers are waiting at the start barrier, so the queue is ours.
            prefill(&queue, workload, mix);
            idle.store(0, Ordering::Relaxed);
            popped.store(0, Ordering::Relaxed);
            let begin = Instant::now();
            start.wait();
            end.wait();
            elapsed += begin.elapsed();
            let left = drain_count(&queue);
            let (expected_left, expected_popped) = match workload {
                Workload::Fan(fan) => (0, fan.total_pops()),
                Workload::Drain => (0, TOTAL_OPS),
                Workload::Fill => (TOTAL_OPS, 0),
            };
            let id = format!("{}/{}/{}/{threads}", workload.name(), mix.name(), Q::NAME);
            assert_eq!(left, expected_left, "{id}: items left in the queue");
            assert_eq!(
                popped.load(Ordering::Relaxed),
                expected_popped,
                "{id}: items popped"
            );
        }
        elapsed
    })
}

/// Checks [`Fan::total_pops`] against a sequential run of the fan-out rule.
fn check_fan_totals() {
    for fan in Fan::ALL {
        let mut stack: Vec<u64> = (0..fan.roots())
            .map(|index| FAN_STEPS << 32 | u64::from(fan.root_low(index)))
            .collect();
        let mut pops = 0;
        while let Some(item) = stack.pop() {
            pops += 1;
            let steps = item >> 32;
            let low = item as u32;
            if steps > 0 {
                for i in 1..=fan.children(low) {
                    stack.push((steps - 1) << 32 | u64::from(low + i));
                }
            }
        }
        assert_eq!(pops, fan.total_pops(), "{}: total pops", fan.name());
        assert!(
            pops <= TOTAL_OPS && pops * 1000 >= TOTAL_OPS * 996,
            "{}: {pops} pops, not within 0.4% of {TOTAL_OPS}",
            fan.name()
        );
    }
}

/// Checks that `uniform128` spreads the `drain`/`fill` items evenly over 128 priorities.
fn check_uniform_priorities() {
    let mut counts = [0usize; 128];
    for index in 0..TOTAL_OPS {
        counts[Mix::Uniform128.priority(item(SETUP_THREAD, index)) as usize] += 1;
    }
    assert!(
        counts.iter().all(|&count| count == TOTAL_OPS / 128),
        "uniform128: uneven priorities"
    );
}

fn bench_impl<Q: BenchQueue>(c: &mut Criterion, threads: usize) {
    let mut group = c.benchmark_group("turbo_concurrent_priority_queue_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for workload in Workload::all() {
        group.throughput(Throughput::Elements(workload.ops() as u64));
        for mix in Mix::ALL {
            let id = BenchmarkId::new(
                format!("{}/{}/{}", workload.name(), mix.name(), Q::NAME),
                threads,
            );
            group.bench_function(id, |b| {
                b.iter_custom(|iters| run::<Q>(workload, mix, threads, iters))
            });
        }
    }
    group.finish();
}

fn scaling(c: &mut Criterion) {
    check_fan_totals();
    check_uniform_priorities();
    // One worker thread per available core; vary the core count from the outside (see the module
    // docs).
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    bench_impl::<ConcurrentPriorityQueue<u32, u64>>(c, threads);
    bench_impl::<MutexHeap>(c, threads);
    bench_impl::<Unordered>(c, threads);
}

criterion_group!(benches, scaling);
criterion_main!(benches);
