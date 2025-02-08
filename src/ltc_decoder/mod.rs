use core::fmt::Display;

use num_traits::{FromPrimitive, ToPrimitive, Zero};

use crate::ltc_decoder::bit_decoder::{BitDecoder, BitVal};
use crate::ltc_frame::LtcFrame;
use crate::TimecodeFrame;

mod bit_decoder;

//pub trait Sample: Copy + Zero + std::ops::Div<f64>+ FromPrimitive + Ord + Sync + Send + 'static {}
//pub trait Sample: Zero + Ord + Clone + Copy + 'static {}

pub trait Sample: Zero + Ord + Clone + Copy + FromPrimitive + ToPrimitive + Display + 'static {}

impl<T> Sample for T where T: Zero + Ord + Clone + Copy + FromPrimitive + ToPrimitive + Display + 'static {}

pub struct LtcDecoder<T: Sample> {
    ltc_frame: LtcFrame,
    bit_decoder: BitDecoder<T>,
    sampling_rate: f32,
}

impl<T: Sample> LtcDecoder<T> {
    pub fn new<S: ToPrimitive>(sampling_rate: S) -> Self {
        Self {
            ltc_frame: LtcFrame::new_empty(),
            bit_decoder: BitDecoder::new(),
            sampling_rate: sampling_rate.to_f32().expect("Invalid sampling rate"),
        }
    }
}

impl<T: Sample> LtcDecoder<T> {
    /// Push received audio-sample-point one after another in this function. From time to time
    /// a Timecode-Frame will be returned to tell the current received timecode
    pub fn get_timecode_frame_for_sample(&mut self, sample: T) -> Option<TimecodeFrame> {
        self.ltc_frame.sample_received(1);
        let bit = self.bit_decoder.get_bit_for_sample(sample);
        self.get_timecode_frame_for_bitval(bit)
    }

    /// Push received audio-sample-point one after another in this function. From time to time
    /// a Timecode-Frame will be returned to tell the current received timecode
    #[deprecated(since="0.4.0", note="please use `get_timecode_frame_for_sample` instead")]
    pub fn get_timecode_frame(&mut self, sample: T) -> Option<TimecodeFrame> {
        self.get_timecode_frame_for_sample(sample)
    }

    /// Alternatively, push received audio sample count (number of samples between zero-crossings)
    /// in this function. From time to time a Timecode-Frame will be returned to tell the current received timecode
    /// 
    /// This function is useful when decoding timecode using a timer peripheral of a microcontroller
    /// to count the number of samples between zero-crossings, rather than decoding each sample
    pub fn get_timecode_frame_for_sample_count(&mut self, sample_count: usize) -> Option<TimecodeFrame> {
        self.ltc_frame.sample_received(sample_count);
        let bit = self.bit_decoder.get_bit_for_sample_count(sample_count);
        self.get_timecode_frame_for_bitval(bit)
    }

    fn get_timecode_frame_for_bitval(&mut self, bit: BitVal) -> Option<TimecodeFrame> {
        match bit {
            BitVal::None => { return None; }
            BitVal::Invalid => {
                self.invalidate();
                return None;
            }
            BitVal::True => { self.ltc_frame.shift_bit(true); }
            BitVal::False => { self.ltc_frame.shift_bit(false); }
        }
        if let Some((data, samples_for_frame)) = self.ltc_frame.get_data() {
            Some(data.make_ltc_frame(self.sample_count_to_duration_s(samples_for_frame)))
        } else {
            None
        }
    }

    fn sample_count_to_duration_s(&self, sample_count: usize) -> f32 {
        (sample_count as f32) / self.sampling_rate
    }

    /// In case some unexpected data is received, this function invalidates the decoder to restart
    /// synchronizing on the heartbeat of the data
    fn invalidate(&mut self) {
        self.ltc_frame.invalidate();
        self.bit_decoder.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Shl;
    use std::fs::File;
    use std::io;
    use std::io::Read;

    use num_traits::Zero;

    use crate::ltc_decoder::{LtcDecoder, Sample};
    use crate::TimecodeFrame;
    use crate::FramesPerSecond::{Thirty, TwentyFive, TwentyFour};

    #[test]
    fn test_sample_trait() {
        test_zero(0_i64);
        test_zero(0_i32);
        test_zero(0_i16);
        test_zero(0_i8);
        test_zero(0_u64);
        test_zero(0_u32);
        test_zero(0_u16);
        test_zero(0_u8);

        test_ord(0_i64);
        test_ord(0_i32);
        test_ord(0_i16);
        test_ord(0_i8);
        test_ord(0_u64);
        test_ord(0_u32);
        test_ord(0_u16);
        test_ord(0_u8);

        test_clone(0_i64);
        test_clone(0_i32);
        test_clone(0_i16);
        test_clone(0_i8);
        test_clone(0_u64);
        test_clone(0_u32);
        test_clone(0_u16);
        test_clone(0_u8);

        test_copy(0_i64);
        test_copy(0_i32);
        test_copy(0_i16);
        test_copy(0_i8);
        test_copy(0_u64);
        test_copy(0_u32);
        test_copy(0_u16);
        test_copy(0_u8);

        test_shl(0_i64);
        test_shl(0_i32);
        test_shl(0_i16);
        test_shl(0_i8);
        test_shl(0_u64);
        test_shl(0_u32);
        test_shl(0_u16);
        test_shl(0_u8);

        test_sample(0_i64);
        test_sample(0_i32);
        test_sample(0_i16);
        test_sample(0_i8);
        test_sample(0_u64);
        test_sample(0_u32);
        test_sample(0_u16);
        test_sample(0_u8);
    }

    fn test_zero<T: Zero>(_s: T) {
        assert!(true);
    }

    fn test_ord<T: Ord>(_s: T) {
        assert!(true);
    }

    fn test_clone<T: Clone>(_s: T) {
        assert!(true);
    }

    fn test_copy<T: Copy>(_s: T) {
        assert!(true);
    }

    fn test_sample<T: Sample>(_s: T) {
        assert!(true);
    }

    fn test_shl<T: Shl>(_s: T) {
        assert!(true);
    }

    #[test]
    fn test_ltc_00100000_2mins_25fps_44100x8() {
        test_timecode_file("testfiles/LTC_00100000_2mins_25fps_44100x8.wav",
                           TimecodeFrame::new(0, 10, 0, 1, TwentyFive),
                           TimecodeFrame::new(0, 12, 1, 0, TwentyFive))
    }


    #[test]
    fn test_ltc_00500000_2mins_30fps_44100x8() {
        test_timecode_file("testfiles/LTC_00500000_2mins_30fps_44100x8.wav",
                           TimecodeFrame::new(0, 50, 0, 1, Thirty),
                           TimecodeFrame::new(0, 52, 1, 0, Thirty))
    }

    #[test]
    fn test_ltc_10000000_2mins_24fps_44100x16() {
        test_timecode_file("testfiles/LTC_10000000_2mins_24fps_44100x16.wav",
                           TimecodeFrame::new(10, 0, 0, 1, TwentyFour),
                           TimecodeFrame::new(10, 2, 1, 0, TwentyFour))
    }

    #[test]
    fn test_ltc_10100000_2mins_25fps_44100x16() {
        test_timecode_file("testfiles/LTC_10100000_2mins_25fps_44100x16.wav",
                           TimecodeFrame::new(10, 10, 0, 1, TwentyFive),
                           TimecodeFrame::new(10, 12, 1, 0, TwentyFive))
    }

    #[test]
    fn test_ltc_10400000_2mins_30fps_44100x16() {
        test_timecode_file("testfiles/LTC_10400000_2mins_30fps_44100x16.wav",
                           TimecodeFrame::new(10, 40, 0, 1, Thirty),
                           TimecodeFrame::new(10, 42, 1, 0, Thirty))
    }

    #[test]
    fn test_ltc_10500000_2mins_24fps_48000x16() {
        test_timecode_file("testfiles/LTC_10500000_2mins_24fps_48000x16.wav",
                           TimecodeFrame::new(10, 50, 0, 1, TwentyFour),
                           TimecodeFrame::new(10, 52, 1, 0, TwentyFour))
    }

    #[test]
    fn test_ltc_11000000_2mins_25fps_48000x16() {
        test_timecode_file("testfiles/LTC_11000000_2mins_25fps_48000x16.wav",
                           TimecodeFrame::new(11, 0, 0, 1, TwentyFive),
                           TimecodeFrame::new(11, 2, 1, 0, TwentyFive))
    }

    #[test]
    fn test_ltc_11300000_2mins_30fps_48000x16() {
        test_timecode_file("testfiles/LTC_11300000_2mins_30fps_48000x16.wav",
                           TimecodeFrame::new(11, 30, 0, 1, Thirty),
                           TimecodeFrame::new(11, 32, 1, 0, Thirty))
    }

    #[test]
    fn test_ltc_11400000_2mins_24fps_44100x16() {
        test_timecode_file("testfiles/LTC_11400000_2mins_24fps_44100x16.wav",
                           TimecodeFrame::new(11, 40, 0, 1, TwentyFour),
                           TimecodeFrame::new(11, 42, 1, 0, TwentyFour))
    }

    #[test]
    fn test_ltc_11500000_2mins_25fps_44100x16() {
        test_timecode_file("testfiles/LTC_11500000_2mins_25fps_44100x16.wav",
                           TimecodeFrame::new(11, 50, 0, 1, TwentyFive),
                           TimecodeFrame::new(11, 52, 1, 0, TwentyFive))
    }

    #[test]
    fn test_ltc_12200000_2mins_30fps_44100x16() {
        test_timecode_file("testfiles/LTC_12200000_2mins_30fps_44100x16.wav",
                           TimecodeFrame::new(12, 20, 0, 1, Thirty),
                           TimecodeFrame::new(12, 22, 1, 0, Thirty))
    }


    /// runs a test on decoding timecode sample by sample with specifing the first expected decoded
    /// Frame (usually 1 frame above the start of the audio, because the lib needs some tim to sync)
    /// and the last expected decoded Frame
    fn test_timecode_file(file: &str, first_tc: TimecodeFrame, last_tc: TimecodeFrame) {
        let mut file = File::open(file).expect("File not found");
        let (sampling_rate, samples) = get_timecode_file_data(&mut file);
        test_timecode_frames(sampling_rate, samples, first_tc, last_tc);
    }

    /// runs a test on decoding timecode sample by sample with specifing the first expected decoded
    /// Frame (usually 1 frame above the start of the audio, because the lib needs some tim to sync)
    /// and the last expected decoded Frame
    fn test_timecode_frames<T: Sample>(sampling_rate: u32, samples: impl Iterator<Item = T>, first_tc: TimecodeFrame, last_tc: TimecodeFrame) {
        let mut decoder = LtcDecoder::<T>::new(sampling_rate);
        let mut timecode = first_tc.clone();
        for sample in samples {
            if let Some(tc) = decoder.get_timecode_frame_for_sample(sample) {
                assert_eq!(tc, timecode);
                timecode.add_frame();
            }
        }
        assert_eq!(timecode, last_tc);
    }

    /// Returns sample rate and data from a wav file that contains timecode data for testing
    fn get_timecode_file_data<R>(file: &mut R) -> (u32, impl Iterator<Item = i32> + use<'_, R>)
        where R: io::Seek + Read, {
        let buf_reader = io::BufReader::with_capacity(1024 * 1024, file); // buffer reads so we don't read one sample at a time
        let reader = hound::WavReader::new(buf_reader).expect("could not open timecode file");
        let spec = reader.spec();
        let data = reader.into_samples::<i32>().step_by(spec.channels as usize).map(|s| s.expect("could not read sample"));
        (spec.sample_rate, data)
    }

}