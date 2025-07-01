// add a button to download large model for the user with one click
// add rustpotter support
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use elevenlabs_rs::ElevenLabsClient;
use elevenlabs_rs::Model;
use elevenlabs_rs::endpoints::genai::tts::{TextToSpeech, TextToSpeechBody};
use futures_util::StreamExt;
use google_ai_rs::{Auth, Client};
use porcupine::PorcupineBuilder;
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};
use serde::Deserialize;
use std::cmp::min;
use std::collections::VecDeque;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::io::Cursor;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use webrtc_vad::{SampleRate, Vad, VadMode};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client as ReqwestClient;

pub struct AudioPlayer {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
}

// Central application context for passing config and handles
pub struct AppContext {
    pub config: Config,
    pub audio_player: AudioPlayer,
    pub porcupine: porcupine::Porcupine,
    pub vad: Mutex<Vad>,
    pub whisper_context: Arc<WhisperContext>,
    pub audio_buffer: Arc<Mutex<VecDeque<i16>>>,
    pub elevenlabs_model: Model,
}

impl AudioPlayer {
    fn new() -> anyhow::Result<Self> {
        let (_stream, stream_handle) = OutputStream::try_default()?;
        Ok(Self {
            _stream,
            stream_handle,
        })
    }
    fn play_sound<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let sink = Sink::try_new(&self.stream_handle)?;
        let file = File::open(path)?;
        let source = Decoder::new(BufReader::new(file))?;
        sink.append(source);
        sink.detach();
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub porcupine_key: String,
    pub gemini_key: String,
    pub elevenlabs_key: String,
    pub whisper_language: String,
    pub wakeword_path: String,
    pub whisper_model_path: String,

    // Advanced
    pub gemini_model: String,
    pub elevenlabs_model: String,
    pub voice_id: String,
    pub llm_system_prompt: String,
    pub vad_mode: String,
    pub wwd_sensitivity: f32,

    pub frame_duration_ms: usize,
    pub silence_threshold_seconds: usize,
    pub speech_trigger_frames: usize,
    pub frame_length_wwd: usize,
}

const SAMPLE_RATE: usize = 16_000;

#[tokio::main]
async fn main() {
    if let Err(e) = run_app().await {
        eprintln!(
            "\n[ERROR] {}\nIf this is your first time running, please check your config.json, model paths, and device setup.\nFor more help, see the README or use --help.\n",
            e
        );
        std::process::exit(1);
    }
}

async fn run_app() -> Result<()> {
    let config = match load_config("assets/config.json") {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!(
                "Failed to load config.json: {}. Please ensure the file exists and is valid JSON.",
                e
            );
            return Err(e.into());
        }
    };

    let audio_player = match AudioPlayer::new() {
        Ok(player) => player,
        Err(e) => {
            eprintln!(
                "Failed to initialize audio output: {}. Please check your audio device.",
                e
            );
            return Err(e.into());
        }
    };

    let porcupine = match PorcupineBuilder::new_with_keyword_paths(
        &config.porcupine_key,
        &[&config.wakeword_path],
    )
    .sensitivities(&[config.wwd_sensitivity])
    .init()
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "Unable to create Porcupine wake word engine: {}. Please check your Porcupine key and wakeword path.",
                e
            );
            return Err(e.into());
        }
    };

    let vad_mode = match config.vad_mode.as_str() {
        "VeryAggressive" => VadMode::VeryAggressive,
        "Aggressive" => VadMode::Aggressive,
        "Normal" => VadMode::LowBitrate,
        "Quality" => VadMode::Quality,
        _ => VadMode::Aggressive,
    };
    let mut vad = Vad::new();
    vad.set_sample_rate(SampleRate::Rate16kHz);
    vad.set_mode(vad_mode);

    let elevenlabs_model = match config.elevenlabs_model.as_str() {
        "eleven_multilingual_v2" => Model::ElevenMultilingualV2,
        "eleven_turbo_v2" => Model::ElevenTurboV2,
        _ => Model::ElevenMultilingualV2,
    };

    let whisper_model_path = Path::new(&config.whisper_model_path);
    if !whisper_model_path.exists() {
        println!(
            "Whisper model not found at '{}'.",
            whisper_model_path.display()
        );
        const WHISPER_MODEL_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin?download=true";
        download_file(WHISPER_MODEL_URL, whisper_model_path).await?;
    }

    let whisper_context = match WhisperContext::new_with_params(
        &config.whisper_model_path,
        WhisperContextParameters::default(),
    ) {
        Ok(ctx) => Arc::new(ctx),
        Err(e) => {
            eprintln!(
                "Failed to load Whisper model: {}. Please check your model path.",
                e
            );
            return Err(e.into());
        }
    };

    let audio_buffer = Arc::new(Mutex::new(VecDeque::<i16>::with_capacity(SAMPLE_RATE * 5)));
    if let Err(e) = std::panic::catch_unwind(|| start_audio_stream(audio_buffer.clone())) {
        eprintln!(
            "Failed to start audio input stream: {e:?}. Please check your microphone device."
        );
        return Err(anyhow::anyhow!(
            "Failed to start audio input stream: {e:?}. Please check your microphone device."
        ));
    }

    let app = AppContext {
        config,
        audio_player,
        porcupine,
        vad: Mutex::new(vad),
        whisper_context,
        audio_buffer,
        elevenlabs_model,
    };

    println!("\n--- Prepared environment successfully  ---");
    clear_console();
    main_loop(&app).await;
    Ok(())
}

async fn main_loop(app: &AppContext) {
    let frame_length_wwd = app.config.frame_length_wwd;
    let frame_duration_ms = app.config.frame_duration_ms;
    let silence_threshold_frames =
        app.config.silence_threshold_seconds * (1000 / frame_duration_ms);
    let speech_trigger_frames = app.config.speech_trigger_frames;
    let whisper_language = &app.config.whisper_language;
    let gemini_key = &app.config.gemini_key;
    let gemini_model = &app.config.gemini_model;
    let voice_id = &app.config.voice_id;
    let llm_system_prompt = &app.config.llm_system_prompt;
    let elevenlabs_key = &app.config.elevenlabs_key;

    loop {
        // Wake Word Detection
        println!("Listening for wake word...");
        loop {
            let frame = next_audio_frame(app.audio_buffer.clone(), frame_length_wwd);
            if let Ok(keyword_index) = app.porcupine.process(&frame) {
                if keyword_index >= 0 {
                    println!("\nWake word detected!");
                    break;
                }
            }
        }
        if let Err(e) = app.audio_player.play_sound("assets/beep.mp3") {
            eprintln!("Failed to play beep sound: {e}");
        }
        println!("Listening for command... (Speak now)");
        let speech_segment =
            record_speech_segment(app, speech_trigger_frames, silence_threshold_frames);

        // STT
        if speech_segment.is_empty() {
            println!("No speech detected after wake word. Please try again.");
        } else {
            println!(
                "Processing {} seconds of audio...",
                speech_segment.len() as f32 / SAMPLE_RATE as f32
            );
            let audio = speech_segment.clone();
            let lang = whisper_language.to_string();
            let gkey = gemini_key.to_string();
            let model = gemini_model.to_string();
            let prompt = llm_system_prompt.to_string();
            let voice = voice_id.to_string();
            let elevenlabs_model_clone = app.elevenlabs_model.clone();
            let whisper_ctx = Arc::clone(&app.whisper_context);
            let ekey = elevenlabs_key.to_string();

            tokio::spawn(async move {
                let user_prompt = transcribe_audio_segment(&whisper_ctx, &audio, &lang);
                let answer = send_to_gemini(&user_prompt, &gkey, &model, &prompt).await;
                let _ = speak_stream(&answer, &voice, elevenlabs_model_clone, &ekey).await;
            });
        }

        println!("\n----------------------------------------\n");
    }
}

// Records a segment of speech using VAD.
fn record_speech_segment(
    app: &AppContext,
    speech_trigger_frames: usize,
    silence_threshold_frames: usize,
) -> Vec<i16> {
    let frame_length_vad = (SAMPLE_RATE / 1000) * app.config.frame_duration_ms;
    let mut is_speaking = false;
    let mut silent_frames = 0;
    let mut speech_frames = 0;
    let mut speech_segment = Vec::new();
    let mut recent_frames: VecDeque<Vec<i16>> = VecDeque::with_capacity(speech_trigger_frames);

    loop {
        let frame = next_audio_frame(app.audio_buffer.clone(), frame_length_vad);
        let mut vad = match app.vad.lock() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to lock VAD mutex: {e}");
                continue;
            }
        };
        let is_speech = match vad.is_voice_segment(&frame) {
            Ok(val) => val,
            Err(e) => {
                eprintln!("VAD processing failed: {:?}", e);
                false
            }
        };
        drop(vad);

        if is_speaking {
            speech_segment.extend_from_slice(&frame);

            if is_speech {
                silent_frames = 0;
                print!(".");
                let _ = std::io::stdout().flush();
            } else {
                silent_frames += 1;
                print!("_");
                let _ = std::io::stdout().flush();

                if silent_frames >= silence_threshold_frames {
                    println!("\nDetected end of speech.");
                    return speech_segment;
                }
            }
        } else if is_speech {
            speech_frames += 1;
            recent_frames.push_back(frame.clone());
            if recent_frames.len() > speech_trigger_frames {
                recent_frames.pop_front();
            }

            if speech_frames >= speech_trigger_frames {
                print!("Speech started: .");
                let _ = std::io::stdout().flush();
                is_speaking = true;
                speech_frames = 0;

                for f in recent_frames.iter() {
                    speech_segment.extend_from_slice(f);
                }

                recent_frames.clear();
            }
        } else {
            speech_frames = 0;
            recent_frames.clear();
        }
    }
}

// Transcribes audio via Whisper.
fn transcribe_audio_segment(
    ctx: &WhisperContext,
    audio_data_i16: &[i16],
    whisper_language: &str,
) -> String {
    let audio_data_f32 = convert_i16_to_f32(audio_data_i16);

    let mut state = match ctx.create_state() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create Whisper state: {e}");
            return String::new();
        }
    };
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(whisper_language));

    if let Err(e) = state.full(params, &audio_data_f32[..]) {
        eprintln!("Failed to run Whisper model: {e}");
        return String::new();
    }

    let num_segments = match state.full_n_segments() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to get number of segments: {e}");
            return String::new();
        }
    };

    let mut full_transcript = String::new();

    println!("\n--- TRANSCRIPTION ---");
    for i in 0..num_segments {
        if let (Ok(segment), Ok(start), Ok(end)) = (
            state.full_get_segment_text(i),
            state.full_get_segment_t0(i),
            state.full_get_segment_t1(i),
        ) {
            let text = segment.trim();
            println!("[{}ms -> {}ms]: {}", start, end, text);
            full_transcript.push_str(text);
            full_transcript.push(' ');
        }
    }
    println!("---------------------\n");

    return full_transcript.clone();
}

async fn send_to_gemini(
    prompt: &str,
    gemini_key: &str,
    gemini_model: &str,
    llm_system_prompt: &str,
) -> String {
    let full_prompt = format!("{}{}", llm_system_prompt, prompt);

    let client = match Client::new(Auth::ApiKey(gemini_key.to_string())).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create Gemini client: {e}");
            return String::from("[Gemini client error]");
        }
    };
    let model = client.generative_model(gemini_model);

    let response = model.generate_content(full_prompt).await;
    let response_text = match response {
        Ok(r) => r.text(),
        Err(e) => {
            eprintln!("Gemini API error: {e}");
            String::from("[Gemini API error]")
        }
    };
    println!(
        "\n--- GEMINI RESPONSE ---\n{}\n------------------------\n",
        response_text
    );
    response_text
}

pub async fn speak_stream(text: &str, voice_id: &str, model: Model, api_key: &str) -> Result<()> {
    let client = ElevenLabsClient::new(api_key.to_string());
    let body = TextToSpeechBody::new(text).with_model_id(model);
    let endpoint = TextToSpeech::new(voice_id.to_string(), body);

    let speech_bytes = match client.hit(endpoint).await {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Failed to get TTS audio from ElevenLabs: {e}");
            return Err(anyhow::anyhow!("TTS error: {e}"));
        }
    };

    let (_stream, stream_handle) = match OutputStream::try_default() {
        Ok(val) => val,
        Err(e) => {
            eprintln!("Failed to get audio output stream: {e}");
            return Err(anyhow::anyhow!("Audio output error: {e}"));
        }
    };
    let sink = match Sink::try_new(&stream_handle) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create audio sink: {e}");
            return Err(anyhow::anyhow!("Audio sink error: {e}"));
        }
    };

    let cursor = Cursor::new(speech_bytes);
    let source = match Decoder::new(cursor) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("Failed to decode audio: {e}");
            return Err(anyhow::anyhow!("Audio decode error: {e}"));
        }
    };
    sink.append(source);
    sink.sleep_until_end();

    Ok(())
}

// Starts the cpal audio input stream in a separate thread.
fn start_audio_stream(buffer: Arc<Mutex<VecDeque<i16>>>) {
    thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                eprintln!("No input device found");
                return;
            }
        };

        let device_name = match device.name() {
            Ok(name) => name,
            Err(e) => {
                eprintln!("Failed to get device name: {e}");
                "Unknown".to_string()
            }
        };
        println!("Using input device: {}", device_name);

        let supported_config = match device.supported_input_configs() {
            Ok(mut configs) => configs.find(|c| {
                c.channels() == 1
                    && c.min_sample_rate().0 <= 16_000
                    && c.max_sample_rate().0 >= 16_000
                    && c.sample_format() == SampleFormat::I16
            }),
            Err(e) => {
                eprintln!("Error getting supported configs: {e}");
                None
            }
        };

        let config = if let Some(c) = supported_config {
            c.with_sample_rate(cpal::SampleRate(16_000))
        } else {
            match device.default_input_config() {
                Ok(cfg) => cfg,
                Err(e) => {
                    eprintln!("No default config: {e}");
                    return;
                }
            }
        };

        println!(
            "Using sample rate: {} Hz, channels: {}, format: {}",
            config.sample_rate().0,
            config.channels(),
            config.sample_format()
        );

        let stream_config: StreamConfig = config.clone().into();
        let err_fn = |err| eprintln!("Stream error: {}", err);
        let channels = stream_config.channels as usize;

        let input_sample_rate = stream_config.sample_rate.0;
        let resample_factor = if input_sample_rate != SAMPLE_RATE as u32 {
            input_sample_rate as f64 / SAMPLE_RATE as f64
        } else {
            1.0
        };

        let mut resample_pos = 0.0;

        let stream = match device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let mut buf = match buffer.lock() {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Failed to lock audio buffer: {e}");
                        return;
                    }
                };

                let samples_iterator: Box<dyn Iterator<Item = i16>> = if resample_factor != 1.0 {
                    let mut resampled = Vec::new();
                    let input_samples = data.iter().step_by(channels).cloned();
                    for sample in input_samples {
                        while resample_pos < 1.0 {
                            resampled.push(sample);
                            resample_pos += resample_factor;
                        }
                        resample_pos -= 1.0;
                    }
                    Box::new(resampled.into_iter())
                } else {
                    Box::new(data.iter().step_by(channels).cloned())
                };

                for sample in samples_iterator {
                    if buf.len() >= buf.capacity() {
                        buf.pop_front();
                    }
                    buf.push_back(sample);
                }
            },
            err_fn,
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to build input stream: {e}");
                panic!("Failed to build input stream");
            }
        };

        if let Err(e) = stream.play() {
            eprintln!("Failed to start input stream: {e}");
            panic!("Failed to start input stream");
        }
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    });
}

fn next_audio_frame(buffer: Arc<Mutex<VecDeque<i16>>>, frame_size: usize) -> Vec<i16> {
    loop {
        let mut buf = match buffer.lock() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to lock audio buffer: {e}");
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        if buf.len() >= frame_size {
            return buf.drain(..frame_size).collect();
        }
        drop(buf);
        thread::sleep(Duration::from_millis(10));
    }
}

// Helper functions
fn clear_console() {
    print!("\x1b[2J\x1b[H\x1b[3J");
    let _ = std::io::stdout().flush();
}

fn convert_i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| s as f32 / 32768.0).collect()
}

fn load_config(path: &str) -> Result<Config> {
    let data = fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(&data)?;
    Ok(config)
}

// --- New Function: Downloads a file with a progress bar ---
async fn download_file(url: &str, path: &Path) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let client = ReqwestClient::new();
    let res = client.get(url).send().await?;
    let total_size = res.content_length().unwrap_or(0);

    // Setup progress bar
    let pb = ProgressBar::new(total_size);
    let style_result = ProgressStyle::with_template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
        .map(|s| s.progress_chars("#>-"));

    if let Ok(style) = style_result {
        pb.set_style(style);
    }

    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    pb.set_message(format!("Downloading {}", file_name));

    let mut file = File::create(path)?;
    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item?;
        file.write_all(&chunk)?;
        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message(format!("Download of {} complete.", file_name));
    Ok(())
}
