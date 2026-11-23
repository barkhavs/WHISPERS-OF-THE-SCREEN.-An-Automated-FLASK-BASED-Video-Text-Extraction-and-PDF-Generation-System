
import logging
import os
import subprocess

import cv2
import paddleocr

from flask import Flask, render_template, request, jsonify, send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)


def create_snapshots_and_ocr_from_youtube_1fps_opencv(
    youtube_url, output_filename, chunk_size
):
    video_path = "temp_youtube_video.mp4"
    cap = None
    pdf_canvas = None

    try:
        # Remove previous temporary video
        if os.path.exists(video_path):
            os.remove(video_path)

        # Download YouTube video
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "-o",
                video_path,
                youtube_url
            ],
            check=True
        )

        # Create output folder
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_folder = os.path.join(desktop_path, "YouTubeSnapshots")
        os.makedirs(output_folder, exist_ok=True)

        # Ensure PDF extension and safe filename
        output_filename = os.path.basename(output_filename)

        if not output_filename.lower().endswith(".pdf"):
            output_filename += ".pdf"

        output_path = os.path.join(output_folder, output_filename)

        # Initialize PDF
        pdf_canvas = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter

        # Open downloaded video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open downloaded video file.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0 or total_frames <= 0:
            raise ValueError("Invalid video FPS or frame count.")

        total_duration = int(total_frames / fps)

        logging.info(f"Video duration: {total_duration} seconds")
        logging.info(f"Video FPS: {fps}")

        # Initialize PaddleOCR 3.x
        ocr = paddleocr.PaddleOCR(
            use_textline_orientation=True,
            lang="en"
        )

        global_snapshot_count = 0

        # Process video in chunks
        for start_time in range(0, total_duration, chunk_size):

            end_time = min(start_time + chunk_size, total_duration)

            logging.debug(
                f"Processing chunk: {start_time} - {end_time} seconds"
            )

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            current_frame = start_frame

            while current_frame < end_frame:

                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                ret, frame = cap.read()

                if not ret:
                    logging.error("Error reading video frame.")
                    break

                snapshot_filename = os.path.join(
                    output_folder,
                    f"snapshot_{global_snapshot_count}.jpg"
                )

                if not cv2.imwrite(snapshot_filename, frame):
                    logging.error(
                        f"Failed to save snapshot: {snapshot_filename}"
                    )
                    current_frame += max(1, int(fps))
                    continue

                # Start a new PDF page after the first snapshot
                if global_snapshot_count > 0:
                    pdf_canvas.showPage()

                # Add snapshot image to PDF
                img = ImageReader(snapshot_filename)

                pdf_canvas.drawImage(
                    img,
                    50,
                    height - 350,
                    width=500,
                    height=300
                )

                # Perform OCR using PaddleOCR 3.x
                result = ocr.predict(input=snapshot_filename)

                ocr_text = ""

                if result:
                    first_result = result[0]

                    # Extract OCR result from PaddleOCR 3.x
                    result_data = getattr(first_result, "json", None)

                    if callable(result_data):
                        result_data = result_data()

                    if isinstance(result_data, dict):
                        result_data = result_data.get(
                            "res", result_data
                        )
                    elif hasattr(first_result, "get"):
                        result_data = first_result

                    if isinstance(result_data, dict):
                        recognized_texts = result_data.get(
                            "rec_texts", []
                        )

                        ocr_text = "\n".join(recognized_texts)

                # Add OCR text to PDF
                if ocr_text.strip():

                    pdf_canvas.setFont("Helvetica-Bold", 11)

                    pdf_canvas.drawString(
                        50,
                        height - 370,
                        "OCR Text:"
                    )

                    pdf_canvas.setFont("Helvetica", 9)

                    text_object = pdf_canvas.beginText(
                        50,
                        height - 390
                    )

                    for line in ocr_text.split("\n"):
                        text_object.textLine(line)

                    pdf_canvas.drawText(text_object)

                else:
                    pdf_canvas.setFont("Helvetica", 10)

                    pdf_canvas.drawString(
                        50,
                        height - 370,
                        "No text detected."
                    )

                global_snapshot_count += 1

                logging.info(
                    f"Processed snapshot {global_snapshot_count}"
                )

                # Move to next second
                current_frame += max(1, int(fps))

        # Release video
        cap.release()
        cap = None

        # Save PDF
        pdf_canvas.save()
        pdf_canvas = None

        logging.info(f"PDF successfully saved: {output_path}")

        # Remove temporary video
        if os.path.exists(video_path):
            os.remove(video_path)

        return output_path

    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading video: {e}", exc_info=True)
        return None

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return None

    finally:
        if cap is not None:
            cap.release()


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_video():

    youtube_url = request.form.get("youtube_url")
    output_filename = request.form.get("pdf")
    chunk_size = request.form.get("cs")

    if not youtube_url:
        return jsonify({"error": "No YouTube URL provided"}), 400

    if not output_filename:
        output_filename = "YouTube_Snapshots.pdf"

    try:
        chunk_size = int(chunk_size)

        if chunk_size <= 0:
            return jsonify({"error": "Chunk size must be greater than zero"}), 400

        pdf_path = create_snapshots_and_ocr_from_youtube_1fps_opencv(
            youtube_url,
            output_filename,
            chunk_size
        )

        if pdf_path and os.path.exists(pdf_path):

            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=os.path.basename(pdf_path)
            )

        return jsonify({
            "error": "Video processing failed. Check terminal logs."
        }), 500

    except (ValueError, TypeError):
        return jsonify({"error": "Please enter a valid chunk size."}), 400

    except Exception as e:
        logging.error(f"Processing route error: {e}", exc_info=True)

        return jsonify({"error": str(e)}), 500


@app.route("/thank_you")
def thank_you():
    return render_template("thank_you.html")


@app.route("/download")
def download_file():

    pdf_path = request.args.get("pdf_path")
    output_filename = request.args.get("output_filename")

    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({"error": "File not found."}), 404

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=output_filename or os.path.basename(pdf_path)
    )


@app.route("/about")
def about():
    return render_template("aboutus.html")


if __name__ == "__main__":
    app.run(debug=True)
