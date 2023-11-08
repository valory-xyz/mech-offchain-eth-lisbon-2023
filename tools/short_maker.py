from typing import Dict, Any, Tuple, Optional

import openai
import json

import requests
from aea_cli_ipfs.ipfs_utils import IPFSTool
from replicate import Client
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

ALLOWED_TOOLS = [
    "short-maker",
]
TOOL_TO_ENGINE = {tool: "gpt-3.5-turbo" for tool in ALLOWED_TOOLS}


def download_file(url: str, local_filename: str):
    """Utility function to download a file from a URL to a local path."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


# Function to send prompts to the OpenAI API
def get_openai_response(user_input: str, engine: str) -> Dict[str, Any]:
    """Construct the message that includes the user input"""
    content =  ( f"Based on the USER INPUT: \"{user_input}\", please provide a narrative script, a prompt for generating a soundtrack, and a prompt for generating a video clip. Format your response as a JSON object with three fields: \"narrative_script\", \"soundtrack_prompt\", and \"video_clip_prompt\". Each field should contain the respective content as described below.\n\n"
                   "- For the \"narrative_script\": Use the user input to create a short script no longer than 10 seconds long, which has a mindblowing insight. The script should be in direct speech format suitable for text-to-speech AI without any stage directions. Do not just repeat the user input\n\n"
                   "- For the \"soundtrack_prompt\": Devise a prompt that would guide an AI to generate a soundtrack that captures the mood implied by the user input.\n\n"
                   "- For the \"video_clip_prompt\": Create a prompt that would guide an AI to generate a video clip that visually corresponds to the user input. The description should be detailed enough to include visually recognizable elements or, if the input is abstract, suggestions for visual representation.\n\n"
                   "Please structure your response in the following JSON format:\n\n"
                   "{\n"
                   "  \"narrative_script\": \"[Insert narrative script here]\",\n"
                   "  \"soundtrack_prompt\": \"[Insert soundtrack prompt here]\",\n"
                   "  \"video_clip_prompt\": \"[Insert video clip prompt here]\"\n"
                   "}\n\n"
                    "Make sure the response can be decoded using json.loads()\n"
                   "EXAMPLE USER INPUT:\n"
                   "This is an example desired response for the user input: \"A group of tiny robots walking down a mountain\"\n\n"
                   "{\n"
                   "  \"narrative_script\": \"In a dance of ingenuity and precision, a swarm of minuscule robots descends a mountain, each step a testament to the colossal power of micro-machinery.\",\n"
                   "  \"soundtrack_prompt\": \"An instrumental soundtrack that blends the sounds of electronic beeps and whirs with a soft, adventurous melody, featuring mechanical and futuristic sounds.\",\n"
                   "  \"video_clip_prompt\": \"A group of small, mechanical robots walking in unison down a rugged mountain path. The robots should appear advanced yet cute, with the mountain landscape providing a vast and wild backdrop. The time of day is dusk, with long shadows and the golden hue of the setting sun giving the scene a warm glow. There should be subtle movements in the robots' parts, suggesting intricate functionality as they navigate the rocky terrain.\"\n"
                   "}\n" )
    message = {
        "role": "system",
        "content": content,
    }

    # Send the message to the chat completions endpoint
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[message]
    )

    # Parse the JSON content from the response
    content = response['choices'][0]['message']['content']

    # Load the content as a JSON object to ensure proper JSON formatting
    json_object = json.loads(content)

    return json_object


def get_voice_over_url(client: Client, text: str) -> str:
    """Narrate the provided text"""
    url = client.run(
      "afiaka87/tortoise-tts:e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71",
      input={
        "seed": 0,
        "text": text,
        "preset": "fast",
        "voice_a": "tom",
        "voice_b": "disabled",
        "voice_c": "disabled"
      }
    )
    return url


def get_soundtrack_url(client: Client, prompt: str, duration: int = 10) -> str:
    """Get a soundtrack for the provided prompt."""
    url = client.run(
        "meta/musicgen:7a76a8258b23fae65c5a22debb8841d1d7e816b75c2f24218cd2bd8573787906",
        input={
            "seed": 3442726813,
            "top_k": 250,
            "top_p": 0,
            "prompt": prompt,
            "duration": duration,
            "temperature": 1,
            "continuation": False,
            "model_version": "large",
            "output_format": "wav",
            "continuation_end": 9,
            "continuation_start": 7,
            "normalization_strategy": "peak",
            "classifier_free_guidance": 3
        }
    )
    return url


def get_video_url(client: Client, prompt: str, num_frames: int = 100, fps: int = 10) -> str:
    """Get a video for the provided prompt."""
    url = client.run(
        "wcarle/text2video-zero-openjourney:2bf28cacd1f02765bd557294ec53f743b42be123675773c810bb3e0f8e3ce6f6",
        input={
            "prompt": prompt,
            "video_length": num_frames,
            "fps": fps
        }
    )
    return url


def combine(voice_over_path: str, soundtrack_path: str, video_path: str) -> str:
    """Combine video, voice over, and soundtrack from provided URLs."""

    # Paths where the downloaded files will be saved
    voice_over_path = 'voice_over.mp3'
    soundtrack_path = 'soundtrack.wav'
    video_path = 'video.mp4'

    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Load the voice-over audio
    voice_over = AudioFileClip(voice_over_path)

    # Load the soundtrack audio
    soundtrack = AudioFileClip(soundtrack_path)

    # Optionally, you can adjust the volume of the voice-over and soundtrack
    voice_over = voice_over.volumex(1.0)  # Adjust volume of voice-over if needed
    soundtrack = soundtrack.volumex(0.25)  # Adjust volume of soundtrack if needed

    # Combine the voice-over and soundtrack, with the voice-over on top
    composite_audio = CompositeAudioClip([soundtrack, voice_over])

    # Set the audio of the video clip to the composite audio clip
    video_clip = video_clip.set_audio(composite_audio)

    # Write the result to a file (the codec and bitrate can be changed as needed)
    output_file = 'output_video.mp4'  # Update with your output path
    video_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

    # Close the clips to release their resources
    video_clip.close()
    voice_over.close()
    soundtrack.close()

    return output_file


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Run the task"""
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    openai_key = kwargs["api_keys"]["openai"]
    openai.api_key = openai_key
    engine = TOOL_TO_ENGINE[tool]

    replicate_key = kwargs["api_keys"]["replicate"]
    client = Client(replicate_key)

    # get the data for all the 3 files
    data = get_openai_response(prompt, engine)

    # get the voice over first, the length of the voice over
    # will dictate the length of the other 2 files
    print(f"Generated the prompt {data}")
    script_prompt = data["narrative_script"]
    voice_over_url = get_voice_over_url(client, script_prompt)
    voice_over_path = "voice_over.mp3"
    download_file(voice_over_url, voice_over_path)
    voice_over = AudioFileClip(voice_over_path)
    print(f"Generated the voiceover")

    # get the video
    fps = 10
    num_frames = int(voice_over.duration * 1.2 * fps)  # give 20% room
    print(num_frames)
    video_path = "video.mp4"
    video_prompt = data["video_clip_prompt"]
    url = get_video_url(client, video_prompt, num_frames, fps)
    download_file(url, video_path)
    print(f"Generated the video")

    # get the bg sound
    soundtrack_prompt = data["soundtrack_prompt"]
    soundtrack_url = get_soundtrack_url(client, soundtrack_prompt, int(voice_over.duration * 1.2))
    soundtrack_path = "soundtrack.wav"
    download_file(soundtrack_url, soundtrack_path)
    print(f"generated the soundtrack")

    output_file = combine(voice_over_path, soundtrack_path, video_path)
    ipfs_tool = IPFSTool()
    _, video_hash_, _ = ipfs_tool.add(output_file, wrap_with_directory=False)

    print(f"stored the output on: {video_hash_}")

    body = {
        "video": video_hash_,
        # todo
        "image": "bafybeig64atqaladigoc3ds4arltdu63wkdrk3gesjfvnfdmz35amv7faq",
    }
    return json.dumps(body), prompt, None