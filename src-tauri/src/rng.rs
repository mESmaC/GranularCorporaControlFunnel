//! Small, dependency-free, seedable RNG (xoshiro256**) so that every sampling
//! decision GCCF makes is reproducible from `(seed, file path)` alone.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut st = seed;
        let s = [
            splitmix64(&mut st),
            splitmix64(&mut st),
            splitmix64(&mut st),
            splitmix64(&mut st),
        ];
        Rng { s }
    }

    /// Derive a child RNG from a seed and an arbitrary string (e.g. a file path).
    pub fn derive(seed: u64, salt: &str) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(&seed.to_le_bytes());
        h.update(salt.as_bytes());
        let out = h.finalize();
        let b = out.as_bytes();
        let mut s = [0u64; 4];
        for (i, chunk) in b.chunks(8).take(4).enumerate() {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            s[i] = u64::from_le_bytes(arr);
        }
        if s.iter().all(|&x| x == 0) {
            return Rng::new(seed);
        }
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `[0, n)`. `n` must be > 0.
    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        // Lemire's nearly-divisionless method.
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let t = n.wrapping_neg() % n;
            while l < t {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }

    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        if v.len() < 2 {
            return;
        }
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }

    /// `k` distinct indices from `[0, n)`, returned sorted ascending.
    pub fn distinct_sorted(&mut self, n: u64, k: usize) -> Vec<u64> {
        let k = (k as u64).min(n) as usize;
        if k == 0 {
            return Vec::new();
        }
        // If we want a large fraction, do a partial Fisher-Yates over a dense range.
        if (k as u64) * 4 >= n && n <= 4_000_000 {
            let mut idx: Vec<u64> = (0..n).collect();
            for i in 0..k {
                let j = i + self.below((n - i as u64) as u64) as usize;
                idx.swap(i, j);
            }
            idx.truncate(k);
            idx.sort_unstable();
            return idx;
        }
        let mut set = std::collections::BTreeSet::new();
        while set.len() < k {
            set.insert(self.below(n));
        }
        set.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn below_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            assert!(r.below(13) < 13);
        }
    }

    #[test]
    fn distinct_sorted_ok() {
        let mut r = Rng::new(1);
        let v = r.distinct_sorted(100, 10);
        assert_eq!(v.len(), 10);
        assert!(v.windows(2).all(|w| w[0] < w[1]));
        let v = r.distinct_sorted(5, 10);
        assert_eq!(v, vec![0, 1, 2, 3, 4]);
        let v = r.distinct_sorted(1_000_000_000, 50);
        assert_eq!(v.len(), 50);
    }
}
