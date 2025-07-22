import json
import os
import logging
import subprocess
import boto3
from PIL import Image, ImageDraw, ImageFont # Pillow is still useful for debugging/fallbacks if needed

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb') # Initialize DynamoDB client

# Get the output bucket name from environment variables
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET")
if not OUTPUT_BUCKET:
    logger.critical("OUTPUT_BUCKET environment variable is not set.")
    raise ValueError("OUTPUT_BUCKET environment variable is not set.")

# --- Configuration for ASL Sign Video Library (YOU WILL CONFIGURE THESE) ---
# This is the S3 bucket where your individual ASL sign video clips are stored.
ASL_SIGN_VIDEO_BUCKET = os.environ.get("ASL_SIGN_VIDEO_BUCKET", "your-asl-sign-videos-bucket") # REPLACE WITH YOUR BUCKET NAME
# This is the DynamoDB table that maps ASL gloss terms to video S3 keys.
ASL_SIGN_MAPPING_TABLE = os.environ.get("ASL_SIGN_MAPPING_TABLE", "ASLSignMapping") # REPLACE WITH YOUR TABLE NAME

# --- Helper to get ASL Gloss from Bedrock (Remains the same) ---
def get_asl_gloss_from_bedrock(english_text):
    logger.info(f"Attempting to get ASL gloss for: '{english_text}' using Bedrock.")
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    prompt_template = """
    You are an expert in American Sign Language (ASL) gloss.
    Translate the following English text into ASL gloss.
    Use uppercase for signs, and use hyphens for multi-word signs.
    For example: "Hello, how are you?" -> "HELLO HOW-ARE-YOU?"
    
    English Text: {text}
    
    ASL Gloss:
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_template.format(text=english_text)}]
            }
        ]
    })
    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get('body').read())
        gloss_text = response_body['content'][0]['text'].strip()
        logger.info(f"Generated ASL Gloss: '{gloss_text}'")
        return gloss_text
    except Exception as e:
        logger.error(f"Error invoking Bedrock for ASL gloss: {e}", exc_info=True)
        return f"ERROR-GLOSS-GENERATION: {english_text}"

# --- NEW FUNCTION: Lookup ASL Sign Video Path (Conceptual) ---
def sign_video_lookup(sign_gloss_term):
    """
    Looks up the S3 key for a given ASL gloss term from DynamoDB.
    
    *** IMPORTANT: You need to implement the DynamoDB table and populate it. ***
    The table should map 'gloss_term' (Partition Key) to 's3_key' (attribute).
    
    Example DynamoDB item:
    {
        "gloss_term": "HELLO",
        "s3_key": "signs/hello.mp4"
    }
    """
    table = dynamodb.Table(ASL_SIGN_MAPPING_TABLE)
    try:
        response = table.get_item(Key={'gloss_term': sign_gloss_term})
        item = response.get('Item')
        if item and 's3_key' in item:
            logger.info(f"Found video for '{sign_gloss_term}': {item['s3_key']}")
            return item['s3_key']
        else:
            logger.warning(f"No video mapping found for gloss term: '{sign_gloss_term}'")
            return None
    except Exception as e:
        logger.error(f"Error looking up sign video in DynamoDB for '{sign_gloss_term}': {e}", exc_info=True)
        return None

# --- NEW FUNCTION: Handle Fingerspelling (Conceptual) ---
def get_fingerspelling_video_paths(word, output_directory):
    """
    Generates a list of video paths for fingerspelling a word.
    
    *** IMPORTANT: You need individual letter videos (e.g., A.mp4, B.mp4). ***
    This is a placeholder that assumes you have 'signs/letters/A.mp4', etc.
    """
    logger.info(f"Fingerspelling fallback for word: '{word}'")
    fingerspelling_paths = []
    for char in word.upper():
        if 'A' <= char <= 'Z':
            # Assuming you have individual video files for each letter
            # e.g., s3://your-asl-sign-videos-bucket/signs/letters/A.mp4
            fingerspelling_paths.append(f"signs/letters/{char}.mp4")
        else:
            logger.warning(f"Skipping non-alphabetic character in fingerspelling: {char}")
    return fingerspelling_paths

# --- Main Lambda Handler ---
def lambda_handler(event, context):
    logger.info(f"Received event for GenerateASLAvatar: {json.dumps(event)}")

    transcribed_text = event.get('transcribedText', 'No transcription provided')
    
    # Log the payload from the previous step (ProcessTranscription) for debugging
    if 'ProcessingResult' in event and 'Payload' in event['ProcessingResult']:
        try:
            processing_payload_str = event['ProcessingResult']['Payload']['body']
            processing_payload = json.loads(processing_payload_str)
            logger.info(f"ProcessingResult Payload from SQS: {processing_payload}")
        except json.JSONDecodeError:
            logger.warning("Could not decode JSON from ProcessTranscriptionFunction payload.")
        except KeyError:
            logger.warning("ProcessTranscriptionFunction payload missing expected keys.")
    
    logger.info(f"Starting ASL avatar generation pipeline for text: '{transcribed_text}'")

    # Phase 1: Get ASL Gloss from Bedrock
    asl_gloss = get_asl_gloss_from_bedrock(transcribed_text)
    if asl_gloss.startswith("ERROR-GLOSS-GENERATION"):
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'ASL Gloss generation failed: {asl_gloss}'})
        }

    # Define paths for FFmpeg and temporary files
    ffmpeg_path = './ffmpeg'
    output_filename = f"asl_avatar_{context.aws_request_id}.mp4"
    output_path = f"/tmp/{output_filename}"
    
    # Directory to store downloaded individual sign videos
    sign_videos_tmp_dir = f"/tmp/sign_videos_{context.aws_request_id}"
    os.makedirs(sign_videos_tmp_dir, exist_ok=True)

    # List to hold paths of local video files to concatenate
    local_video_paths_to_stitch = []
    
    try:
        # Phase 2: Parse ASL Gloss, Look up Signs, Download Videos
        # Split gloss into individual terms (handle multi-word signs with hyphens)
        gloss_terms = asl_gloss.replace('ASL Gloss:', '').strip().split()
        
        for term in gloss_terms:
            # Attempt to look up the sign video
            s3_key = sign_video_lookup(term.upper()) # Convert to uppercase for lookup consistency
            
            if s3_key:
                # Download the video from S3
                local_video_path = os.path.join(sign_videos_tmp_dir, os.path.basename(s3_key))
                logger.info(f"Downloading s3://{ASL_SIGN_VIDEO_BUCKET}/{s3_key} to {local_video_path}")
                s3_client.download_file(ASL_SIGN_VIDEO_BUCKET, s3_key, local_video_path)
                local_video_paths_to_stitch.append(local_video_path)
            else:
                # Fallback to fingerspelling if sign not found
                logger.warning(f"Sign '{term}' not found in mapping. Attempting fingerspelling.")
                fingerspelling_s3_keys = get_fingerspelling_video_paths(term, sign_videos_tmp_dir)
                for fs_s3_key in fingerspelling_s3_keys:
                    fs_local_path = os.path.join(sign_videos_tmp_dir, os.path.basename(fs_s3_key))
                    logger.info(f"Downloading fingerspelling s3://{ASL_SIGN_VIDEO_BUCKET}/{fs_s3_key} to {fs_local_path}")
                    s3_client.download_file(ASL_SIGN_VIDEO_BUCKET, fs_s3_key, fs_local_path)
                    local_video_paths_to_stitch.append(fs_local_path)

        if not local_video_paths_to_stitch:
            logger.error("No ASL sign videos or fingerspelling videos were found/generated.")
            return {
                'statusCode': 500,
                'body': json.dumps({'message': 'ASL video generation failed: No video segments found.'})
            }

        # Create a file list for FFmpeg concat demuxer
        concat_list_path = os.path.join("/tmp", f"concat_list_{context.aws_request_id}.txt")
        with open(concat_list_path, "w") as f:
            for path in local_video_paths_to_stitch:
                f.write(f"file '{path}'\n")
        logger.info(f"FFmpeg concat list created at {concat_list_path}")

        # Step 3: Use FFmpeg to stitch the downloaded video segments
        # Using concat demuxer for seamless stitching of multiple videos
        ffmpeg_command = [
            ffmpeg_path,
            '-y', # Overwrite output files without asking
            '-f', 'concat',
            '-safe', '0', # Required for file paths in concat list
            '-i', concat_list_path,
            '-c', 'copy', # Copy streams directly (no re-encoding for speed/quality)
            output_path
        ]

        logger.info(f"Running FFmpeg command to stitch videos: {' '.join(ffmpeg_command)}")
        result = subprocess.run(ffmpeg_command, capture_output=True, check=True)
        logger.info(f"FFmpeg stdout (stitching): {result.stdout.decode()}")
        logger.info(f"FFmpeg stderr (stitching): {result.stderr.decode()}")
        logger.info(f"Final ASL video generated at {output_path}")

        # Step 4: Upload generated video to S3
        s3_key = f"output/asl_avatar_{context.aws_request_id}.mp4"
        s3_client.upload_file(output_path, OUTPUT_BUCKET, s3_key)
        logger.info(f"Final ASL video uploaded to s3://{OUTPUT_BUCKET}/{s3_key}")

        video_url = f"https://{OUTPUT_BUCKET}.s3.amazonaws.com/{s3_key}"
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'ASL avatar generated and uploaded successfully.',
                'transcribedText': transcribed_text,
                'aslGloss': asl_gloss,
                'outputVideoUrl': video_url
            })
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed during video stitching: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Video stitching failed: {e}'})
        }
    except Exception as e:
        logger.error(f"Error during ASL video generation or S3 upload: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Internal server error during ASL avatar generation: {e}'})
        }
    finally:
        # Clean up temporary files and directories
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Cleaned up temporary final video file: {output_path}")
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
            logger.info(f"Cleaned up temporary concat list: {concat_list_path}")
        if os.path.exists(sign_videos_tmp_dir):
            # Remove directory and its contents
            for item in os.listdir(sign_videos_tmp_dir):
                item_path = os.path.join(sign_videos_tmp_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            os.rmdir(sign_videos_tmp_dir)
            logger.info(f"Cleaned up temporary sign videos directory: {sign_videos_tmp_dir}")

