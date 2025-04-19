import asyncio
import websockets
import json
import base64
from pydub import AudioSegment
import time
import random


async def connect_to_api(audio_list, json_output_path, temperature, max_retries=100, initial_retry_delay=5):
    api_key = ""  # Replace with your actual API key
    url = "wss://api.openai.com/v1/realtime"

    # List to store all messages
    all_messages = []
    retries = 0
    retry_delay = initial_retry_delay

    while retries < max_retries:
        asyncio_timeoff = False
        try:
            async with websockets.connect(url, subprotocols=["realtime", f"websocket.api_key.{api_key}"]) as websocket:

                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=30)

                        # Save the response to the list
                        all_messages.append(json.loads(response))

                        # Parse the response
                        message = json.loads(response)

                        # Handle different event types
                        if message["event"] == "start_session":
                            await send_config(websocket, temperature)
                            await send_test_message(websocket, audio_list)
                        elif message["event"] == "turn_finished":
                            break  # Exit the loop when "turn_finished" event is received
                        elif message["event"] == "error":
                            print(f"Error received: {message.get('error', 'Unknown error')}")
                            break
                        # Add more event handlers as needed

                    except asyncio.TimeoutError:
                        # print("[VirtueAI] No message received within timeout period. Sending keep-alive...")
                        # await send_keep_alive(websocket)
                        print("[VirtueAI] No message received within timeout period. Re-send the message.")
                        asyncio_timeoff = True
                        break

            if asyncio_timeoff:
                continue

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"[VirtueAI] Connection closed with error: {e}")
            retries += 1
            print(f"[VirtueAI] Retrying... ({retries}/{max_retries}) after {retry_delay} seconds")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
            if retry_delay > 512:
                retry_delay = random.randint(0, 2)
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"[VirtueAI] Invalid status code error: {e}")
            retries += 1
            print(f"[VirtueAI] Retrying... ({retries}/{max_retries}) after {retry_delay} seconds")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
            if retry_delay > 512:
                retry_delay = random.randint(0, 2)
        except Exception as e:
            print(f"[VirtueAI] An unexpected error occurred: {e}")
            retries += 1
            print(f"[VirtueAI] Retrying... ({retries}/{max_retries}) after {retry_delay} seconds")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
            if retry_delay > 512:
                retry_delay = random.randint(0, 2)
        else:
            break  # Exit loop if connection succeeds

    with open(json_output_path, "w") as f:
        json.dump(all_messages, f, indent=4)

    if retries == max_retries:
        raise Exception("Max retries reached. Exiting.")


async def send_config(websocket, temperature):
    config_message = {
        "event": "set_inference_config",
        "system_message": "You are a helpful assistant",
        "turn_end_type": "client_decision",
        "voice": "alloy",
        "audio_format": "pcm16",
        "temperature": temperature,
        "max_tokens": 2048
    }
    await websocket.send(json.dumps(config_message))


async def send_test_message(websocket, audio_file_path_list):
    # Demonstration:
    #
    # audio_file_path_list = [
    #     "./audio_instructions/instruction_general.wav",
    #     "./audio_instructions/instruction_first.wav",
    #     audio1,
    #     "./audio_instructions/instruction_second.wav",
    #     audio2,
    #     "./audio_instructions/instruction_Q_intro.wav",
    #     # f"./audio_instructions/{ins_name}",
    #     ins_name,
    #     "./audio_instructions/instruction_Q_choice.wav",
    # ]
    audio_file_list = []

    for path in audio_file_path_list:
        audio_file_list.append(AudioSegment.from_file(path))

    # Combine the audio files
    combined_audio = sum(audio_file_list)

    # Convert the combined audio to the required format and message
    audio_message = audio_to_msg(combined_audio.raw_data)
    await websocket.send(audio_message)

    # After sending the message, indicate that the client turn is finished
    await websocket.send(json.dumps({"event": "client_turn_finished"}))


async def send_keep_alive(websocket):
    # Send a minimal audio buffer to keep the connection alive
    keep_alive_message = {
        "event": "audio_buffer_add",
        "data": base64.b64encode(b'\x00' * 100).decode()  # 100 bytes of silence
    }
    await websocket.send(json.dumps(keep_alive_message))


def audio_to_msg(audio_data: bytes) -> str:
    # Encode to base64 string
    pcm_base64 = base64.b64encode(audio_data).decode()

    # Create the message in the required format
    msg = {"event": "audio_buffer_add", "data": pcm_base64}
    return json.dumps(msg)


def audioCall(audio_list, json_output_path, temperature=0.8):
    asyncio.get_event_loop().run_until_complete(connect_to_api(audio_list, json_output_path, temperature))