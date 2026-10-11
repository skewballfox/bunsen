//! A sliding buffer of log-mel frames.

use burn::Tensor;

/// The frames of one stream from the seek pointer onward, in a buffer that is
/// written in place.
///
/// The live frames are `buf[head .. head + len]` along dim 1. A push writes
/// after the last live frame with `slice_assign`, which runs in place while
/// the ring owns the only handle to the buffer. Dropping frames from the front
/// moves `head` and copies nothing. Only when a push reaches the end of the
/// buffer are the live frames copied, to the start of a new buffer, and the
/// buffer doubles if they fill more than half of it. So each frame is copied
/// at most once more on average, however the stream is chunked; appending by
/// `cat` copies the whole ring on every push.
///
/// The tail of the buffer past the live frames is uninitialized, and nothing
/// reads it.
#[derive(Clone, Debug)]
pub(crate) struct FrameRing {
    /// `[1, capacity, n_mels]`; `None` until the first push.
    buf: Option<Tensor<3>>,

    /// The buffer index of the first live frame.
    head: usize,

    /// The number of live frames.
    len: usize,

    /// The capacity of the first buffer, in frames.
    min_capacity: usize,
}

impl FrameRing {
    /// An empty ring; its first buffer holds at least `min_capacity` frames.
    pub(crate) fn new(min_capacity: usize) -> Self {
        Self {
            buf: None,
            head: 0,
            len: 0,
            min_capacity: min_capacity.max(1),
        }
    }

    /// The number of live frames.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Appends `[1, count, n_mels]` frames after the last live frame.
    pub(crate) fn push(
        &mut self,
        frames: Tensor<3>,
    ) {
        let [_, count, n_mels] = frames.dims();
        if count == 0 {
            return;
        }

        let capacity = self.buf.as_ref().map_or(0, |buf| buf.dims()[1]);
        if self.head + self.len + count > capacity {
            let needed = self.len + count;
            let capacity = if needed * 2 <= capacity {
                capacity
            } else {
                (capacity * 2).max(needed).max(self.min_capacity)
            };
            self.reallocate(capacity, n_mels, &frames);
        }

        let at = self.head + self.len;
        let buf = self.buf.take().expect("reallocate made a buffer");
        self.buf = Some(buf.slice_assign([0..1, at..at + count, 0..n_mels], frames));
        self.len += count;
    }

    /// Drops the first `count` live frames, or all of them if there are
    /// fewer.
    pub(crate) fn drop_front(
        &mut self,
        count: usize,
    ) {
        let count = count.min(self.len);
        self.head += count;
        self.len -= count;
    }

    /// Drops the last `count` live frames, or all of them if there are fewer.
    pub(crate) fn drop_back(
        &mut self,
        count: usize,
    ) {
        self.len -= count.min(self.len);
    }

    /// The live frames `start .. start + count`: `[1, count, n_mels]`.
    ///
    /// A view of the buffer, not a copy. Drop it before the next
    /// [`push`](Self::push), or that push copies the buffer.
    ///
    /// # Panics
    /// If the range is not inside the live frames.
    pub(crate) fn window(
        &self,
        start: usize,
        count: usize,
    ) -> Tensor<3> {
        assert!(
            start + count <= self.len,
            "window {start}..{} is past the {} live frames",
            start + count,
            self.len,
        );
        let at = self.head + start;
        self.buf
            .as_ref()
            .expect("live frames have a buffer")
            .clone()
            .slice_dim(1, at..at + count)
    }

    /// Moves the live frames to the start of a new buffer of `capacity`
    /// frames, with the dtype and device of `source`.
    fn reallocate(
        &mut self,
        capacity: usize,
        n_mels: usize,
        source: &Tensor<3>,
    ) {
        let fresh = Tensor::empty([1, capacity, n_mels], (&source.device(), source.dtype()));
        self.buf = Some(match self.buf.take() {
            Some(old) if self.len > 0 => {
                let kept = old.slice_dim(1, self.head..self.head + self.len);
                fresh.slice_assign([0..1, 0..self.len, 0..n_mels], kept)
            }
            _ => fresh,
        });
        self.head = 0;
    }
}

#[cfg(test)]
mod test {
    use burn::{
        Tensor,
        prelude::TensorData,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    /// `[1, count, 2]` frames whose values are their stream index.
    fn frames(
        from: usize,
        count: usize,
    ) -> Tensor<3> {
        let values: Vec<f32> = (from..from + count)
            .flat_map(|i| [i as f32, i as f32])
            .collect();
        Tensor::from_data(TensorData::new(values, [1, count, 2]), &cpu_device())
    }

    fn indices(window: Tensor<3>) -> Vec<f32> {
        window
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .step_by(2)
            .collect()
    }

    #[test]
    fn test_push_and_window() {
        let mut ring = FrameRing::new(4);
        ring.push(frames(0, 3));
        ring.push(frames(3, 2));

        assert_eq!(ring.len(), 5);
        assert_eq!(indices(ring.window(0, 5)), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(indices(ring.window(2, 2)), vec![2.0, 3.0]);
    }

    #[test]
    fn test_drop_front_then_push_compacts() {
        let mut ring = FrameRing::new(4);
        let mut next = 0;
        let mut first = 0;
        // Many more frames than the capacity: the ring compacts and grows,
        // and always reads back the live frames in order.
        for step in 0..40 {
            let count = step % 3 + 1;
            ring.push(frames(next, count));
            next += count;

            let drop = (step % 4).min(ring.len());
            ring.drop_front(drop);
            first += drop;

            let expected: Vec<f32> = (first..next).map(|i| i as f32).collect();
            assert_eq!(ring.len(), next - first);
            if ring.len() > 0 {
                assert_eq!(indices(ring.window(0, ring.len())), expected);
            }
        }
    }

    #[test]
    fn test_drop_back() {
        let mut ring = FrameRing::new(4);
        ring.push(frames(0, 3));
        ring.drop_back(1);
        assert_eq!(indices(ring.window(0, ring.len())), vec![0.0, 1.0]);

        ring.drop_back(5);
        assert_eq!(ring.len(), 0);
    }

    #[test]
    #[should_panic(expected = "past the")]
    fn test_window_past_the_end_panics() {
        let mut ring = FrameRing::new(4);
        ring.push(frames(0, 2));
        let _ = ring.window(1, 2);
    }
}
