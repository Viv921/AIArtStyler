# AIArtStyler

AIArtStyler is a full-stack web application designed to generate and classify artistic images. It combines a powerful Python-based AI backend using diffusion models with a modern React and TypeScript frontend.

Users can upload an image to classify its art style or generate new images from a text prompt, guiding the generation with either a specific style name or an uploaded style reference image.

-----

## Features

  * **Image Generation:** Create new images from a text prompt.
      * Use a reference image to guide the style.
      * Use a predefined style name.
      * Control advanced parameters like negative prompts, image dimensions, steps, and guidance scale.
  * **Style Classification:** Upload an image to classify its artistic style, returning the top 3 predicted styles and their confidence scores.
  * **Image Gallery:** A built-in gallery browses previously generated images, sorted by creation date.

-----

## Tech Stack

The project is divided into two main parts: a Python backend and a React frontend.

| Backend | Frontend |
| :--- | :--- |
| Python | React |
| FastAPI | TypeScript |
| Uvicorn | Vite |
| PyTorch | Tailwind CSS |
| Diffusers | Radix UI (for components) |
| Scikit-learn | Lucide Icons (lucide-react) |

-----

## Installation and Setup

To run this project locally, you must set up both the backend and frontend services.

### Prerequisites

  * Python 3.8+ and `pip`
  * Node.js (v18 or later recommended) and `npm`

### 1\. Backend Setup (Python)

The backend server handles all AI operations and file management.

1.  **Clone the repository** (if you haven't already):

    ```sh
    git clone https://github.com/viv921/aiartstyler.git
    cd aiartstyler
    ```

2.  **Create and activate a virtual environment** (recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies** from `requirements.txt`:

    ```sh
    pip install -r requirements.txt
    ```

    **Note:** The `torch` dependency may require a specific installation command based on your hardware (e.g., CUDA version). If the standard install fails, please see the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command for your system.

### 2\. Frontend Setup (Node.js)

The frontend provides the user interface in the browser.

1.  **Navigate to the frontend directory:**

    ```sh
    cd frontend
    ```

2.  **Install Node.js dependencies** from `package.json`:

    ```sh
    npm install
    ```

3.  **Return to the project root directory:**

    ```sh
    cd ..
    ```

-----

## Running the Application

You will need to run two separate terminal commands to start both the backend and frontend.

### 1\. Start the Backend Server

The FastAPI server runs using `uvicorn`.

1.  **From the project's root directory** (e.g., `aiartstyler/`), run:
    ```sh
    uvicorn src.api:app --host 127.0.0.1 --port 8080 --reload
    ```
2.  The backend API is now running at `http://127.0.0.1:8080`.
      * The `--reload` flag will automatically restart the server when you make changes to the Python code.

### 2\. Start the Frontend Server

The React app runs using a Vite development server.

1.  **Open a new terminal window.**
2.  **Navigate to the `frontend` directory:**
    ```sh
    cd frontend
    ```
3.  **Start the Vite dev server** using the script from `package.json`:
    ```sh
    npm run dev
    ```
4.  The frontend is now running and accessible at `http://localhost:5173`.
      * The backend is configured to accept requests from this origin. You can now open this URL in your browser to use the application.

-----

## API Endpoints

The FastAPI backend provides the following main endpoints:

  * `GET /`: Confirms that the API is running.
  * `POST /classify/`: Upload an image file to classify its style.
      * **Body:** `image` (file)
  * `POST /generate/`: Generate an image based on a prompt and style.
      * **Body:** `prompt` (form data), and *either* `style_image` (file) *or* `style_name` (form data). Also accepts advanced parameters like `width`, `height`, `seed`, etc.
  * `GET /gallery/`: Returns a JSON list of all generated images and their URLs.
  * `GET /outputs/{filename}`: Serves a specific image file from the output gallery.

-----

## Project Structure

```
aiartstyler/
├── frontend/           # React + Vite frontend application
│   ├── src/            # Frontend source code (components, styles)
│   ├── package.json    # Frontend dependencies and scripts
│   └── README.md       # Original frontend README
├── models/             # Contains trained classifier models (.pth)
├── src/                # Backend Python source code
│   ├── api.py          # FastAPI endpoints and server logic
│   ├── generator.py    # Core AI logic (generation, classification)
│   ├── config.py       # Configuration variables
│   └── ...             # Other helper modules (dataset, train, etc.)
├── temp_uploads/       # Temporary storage for uploaded files
├── outputs/            # Default gallery location for generated images
├── requirements.txt    # Backend Python dependencies
└── README.md           # This file
```
