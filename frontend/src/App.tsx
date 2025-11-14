import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Spinner } from "@/components/ui/spinner"
import { Paintbrush, Image as ImageIcon, GalleryVertical } from "lucide-react"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChevronDown } from "lucide-react"

// --- Constants ---
const API_BASE_URL = "http://127.0.0.1:8080"

// --- Type Definitions ---
type Page = "detect" | "generate" | "gallery"

type StylePrediction = {
  style: string
  confidence: number
}

type GalleryImage = {
  filename: string
  url: string
}

// --- Main App Component ---
function App() {
  const [page, setPage] = useState<Page>("detect")
  const [detectedStyle, setDetectedStyle] = useState<string | null>(null)

  // This function allows the detector page to send data to the generator page
  const handleStyleDetected = (style: string) => {
    setDetectedStyle(style)
    setPage("generate") // Switch to the generate page
  }

  return (
    <div className="flex flex-col items-center min-h-screen bg-background text-foreground p-4 md:p-8">
      <header className="w-full max-w-4xl mb-8">
        <h1 className="text-3xl font-bold text-center mb-2">🎨 AI Art Stylizer</h1>
        <p className="text-center text-muted-foreground mb-6">
          Detect, Generate, and Browse AI-powered art styles.
        </p>

        {/* Main Navigation */}
        <Tabs value={page} onValueChange={(value) => setPage(value as Page)} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="detect">
              <Paintbrush className="mr-2 h-4 w-4" /> Detect Style
            </TabsTrigger>
            <TabsTrigger value="generate">
              <ImageIcon className="mr-2 h-4 w-4" /> Generate Art
            </TabsTrigger>
            <TabsTrigger value="gallery">
              <GalleryVertical className="mr-2 h-4 w-4" /> Gallery
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </header>

      <main className="w-full max-w-4xl">
        {page === "detect" && <StyleDetectorPage onStyleDetected={handleStyleDetected} />}
        {page === "generate" && <GeneratorPage initialStyle={detectedStyle} />}
        {page === "gallery" && <GalleryPage />}
      </main>
    </div>
  )
}

// --- 1. Style Detector Page ---
function StyleDetectorPage({ onStyleDetected }: { onStyleDetected: (style: string) => void }) {
  const [image, setImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<StylePrediction[] | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setImage(file)
      setImagePreview(URL.createObjectURL(file))
      setResults(null) // Clear old results
      setError(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!image) {
      setError("Please upload an image first.")
      return
    }

    setIsLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append("image", image)

    try {
      const response = await fetch(`${API_BASE_URL}/classify/`, {
        method: "POST",
        body: formData,
      })
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || "Classification failed.")
      }
      const data: StylePrediction[] = await response.json()
      setResults(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Style Detector</CardTitle>
        <CardDescription>Upload an image to identify its artistic style.</CardDescription>
      </CardHeader>
      <form onSubmit={handleSubmit}>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="style-image">Upload Image</Label>
            <Input id="style-image" type="file" accept="image/*" onChange={handleFileChange} />
          </div>
          {imagePreview && (
            <div className="flex justify-center">
              <img src={imagePreview} alt="Style preview" className="rounded-md object-contain max-h-60" />
            </div>
          )}
          {error && <p className="text-destructive text-sm">{error}</p>}
        </CardContent>
        <CardFooter>
          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? <Spinner className="mr-2 h-4 w-4" /> : null}
            {isLoading ? "Analyzing..." : "Detect Style"}
          </Button>
        </CardFooter>
      </form>

      {results && (
        <CardContent className="space-y-4">
          <h3 className="font-semibold">Detection Results:</h3>
          <div className="space-y-3">
            {results.map((result, index) => (
              <div key={result.style} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className={index === 0 ? "font-bold" : ""}>{result.style.replace(/_/g, " ")}</span>
                  <span className="text-muted-foreground">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <Progress value={result.confidence * 100} />
              </div>
            ))}
          </div>
          <Button
            variant="outline"
            className="w-full"
            onClick={() => onStyleDetected(results[0].style)}
          >
            Use "{results[0].style.replace(/_/g, " ")}" to Generate
          </Button>
        </CardContent>
      )}
    </Card>
  )
}

// --- 2. Generator Page (REPLACE this whole function) ---
function GeneratorPage({ initialStyle }: { initialStyle: string | null }) {
  const [prompt, setPrompt] = useState("")
  const [styleImage, setStyleImage] = useState<File | null>(null)
  const [styleName, setStyleName] = useState("")
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [generationMode, setGenerationMode] = useState<"image" | "name">("image")

  // --- NEW: Advanced Options State ---
  const [negativePrompt, setNegativePrompt] = useState("")
  const [negativePrompt2, setNegativePrompt2] = useState("")
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [steps, setSteps] = useState(50)
  const [guidance, setGuidance] = useState(7.5)
  const [seed, setSeed] = useState("-1") // Use string for input, -1 for random

  // This effect listens for the `initialStyle` prop from the detector page
  useEffect(() => {
    if (initialStyle) {
      setStyleName(initialStyle)
      setGenerationMode("name") // Switch to the "Style Name" tab
    }
  }, [initialStyle])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setStyleImage(file)
      setImagePreview(URL.createObjectURL(file))
    } else {
      setStyleImage(null)
      setImagePreview(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!prompt) {
      setError("Please enter a prompt.")
      return
    }
    if (generationMode === "image" && !styleImage) {
      setError("Please upload a style image.")
      return
    }
    if (generationMode === "name" && !styleName) {
      setError("Please enter a style name.")
      return
    }

    setIsLoading(true)
    setError(null)
    setGeneratedImage(null)

    const formData = new FormData()

    // --- Add ALL fields to FormData ---
    formData.append("prompt", prompt)

    // Basic fields
    if (generationMode === "image" && styleImage) {
      formData.append("style_image", styleImage)
    } else if (generationMode === "name" && styleName) {
      formData.append("style_name", styleName)
    }

    // Advanced fields
    if (negativePrompt) formData.append("negative_prompt", negativePrompt)
    if (negativePrompt2) formData.append("negative_prompt_2", negativePrompt2)
    formData.append("width", width.toString())
    formData.append("height", height.toString())
    formData.append("num_inference_steps", steps.toString())
    formData.append("guidance_scale", guidance.toString())
    formData.append("seed", seed)

    try {
      const response = await fetch(`${API_BASE_URL}/generate/`, {
        method: "POST",
        body: formData,
      })
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || "Generation failed.")
      }
      const imageBlob = await response.blob()
      const imageObjectURL = URL.createObjectURL(imageBlob)
      setGeneratedImage(imageObjectURL)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  // Helper for size selects
  const handleSizeChange = (value: string) => {
    const [w, h] = value.split("x").map(Number);
    setWidth(w);
    setHeight(h);
  };
  const sizeValue = `${width}x${height}`;

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Generate Art</CardTitle>
        <CardDescription>Create a new image by combining a content prompt with a style.</CardDescription>
      </CardHeader>

      <Tabs value={generationMode} onValueChange={(v) => setGenerationMode(v as "image" | "name")} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="image">Use Style Image</TabsTrigger>
          <TabsTrigger value="name">Use Style Name</TabsTrigger>
        </TabsList>

        <form onSubmit={handleSubmit}>
          {/* --- Main prompt inputs --- */}
          <TabsContent value="image" className="p-6 space-y-4">
            <div className="space-y-2">
              <Label htmlFor="style-image-gen">1. Upload Style Image</Label>
              <Input id="style-image-gen" type="file" accept="image/*" onChange={handleFileChange} />
            </div>
            {imagePreview && (
              <div className="flex justify-center">
                <img src={imagePreview} alt="Style preview" className="rounded-md object-contain max-h-40" />
              </div>
            )}
            <div className="space-y-2">
              <Label htmlFor="prompt-image">2. Enter Content Prompt</Label>
              <Input id="prompt-image" type="text" placeholder="e.g., 'A cat playing a piano'" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </div>
          </TabsContent>

          <TabsContent value="name" className="p-6 space-y-4">
            <div className="space-y-2">
              <Label htmlFor="style-name-gen">1. Enter Style Name</Label>
              <Input id="style-name-gen" type="text" placeholder="e.g., 'Cubism', 'Anime'" value={styleName} onChange={(e) => setStyleName(e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="prompt-name">2. Enter Content Prompt</Label>
              <Input id="prompt-name" type="text" placeholder="e.g., 'A cat playing a piano'" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </div>
          </TabsContent>

          {/* --- NEW: Advanced Options Accordion --- */}
          <Accordion type="single" collapsible className="w-full px-6">
            <AccordionItem value="item-1">
              <AccordionTrigger>
                <div className="flex items-center text-sm font-medium">
                  <ChevronDown className="mr-2 h-4 w-4 shrink-0 transition-transform duration-200" />
                  Advanced Options
                </div>
              </AccordionTrigger>
              <AccordionContent className="space-y-6 pt-4">

                {/* Negative Prompts */}
                <div className="space-y-2">
                  <Label htmlFor="neg-prompt">Negative Prompt</Label>
                  <Input id="neg-prompt" type="text" placeholder="e.g., ugly, blurry, text, watermark" value={negativePrompt} onChange={(e) => setNegativePrompt(e.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="neg-prompt-2">Negative Prompt 2 (Style)</Label>
                  <Input id="neg-prompt-2" type="text" placeholder="e.g., worst quality, low quality" value={negativePrompt2} onChange={(e) => setNegativePrompt2(e.target.value)} />
                </div>

                {/* Size */}
                <div className="space-y-2">
                  <Label>Image Dimensions</Label>
                  <Select onValueChange={handleSizeChange} value={sizeValue}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select size" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1024x1024">1024 x 1024 (Square)</SelectItem>
                      <SelectItem value="1152x896">1152 x 896 (Landscape 9:7)</SelectItem>
                      <SelectItem value="896x1152">896 x 1152 (Portrait 7:9)</SelectItem>
                      <SelectItem value="1216x832">1216 x 832 (Landscape 19:13)</SelectItem>
                      <SelectItem value="832x1216">832 x 1216 (Portrait 13:19)</SelectItem>
                      <SelectItem value="1344x768">1344 x 768 (Landscape 7:4)</SelectItem>
                      <SelectItem value="768x1344">768 x 1344 (Portrait 4:7)</SelectItem>
                      <SelectItem value="1536x640">1536 x 640 (Landscape 12:5)</SelectItem>
                      <SelectItem value="640x1536">640 x 1536 (Portrait 5:12)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Sliders */}
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Steps</Label>
                    <span className="text-sm text-muted-foreground">{steps}</span>
                  </div>
                  <Slider value={[steps]} min={10} max={100} step={1} onValueChange={(val) => setSteps(val[0])} />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Guidance Scale (CFG)</Label>
                    <span className="text-sm text-muted-foreground">{guidance.toFixed(1)}</span>
                  </div>
                  <Slider value={[guidance]} min={1} max={20} step={0.5} onValueChange={(val) => setGuidance(val[0])} />
                </div>

                {/* Seed */}
                <div className="space-y-2">
                  <Label htmlFor="seed">Seed</Label>
                  <Input id="seed" type="text" placeholder="e.g., 12345 (or -1 for random)" value={seed} onChange={(e) => setSeed(e.target.value)} />
                </div>

              </AccordionContent>
            </AccordionItem>
          </Accordion>

          {error && <p className="text-destructive text-sm px-6 pt-4">{error}</p>}

          <CardFooter className="pt-6">
            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? <Spinner className="mr-2 h-4 w-4" /> : null}
              {isLoading ? "Generating..." : "Generate"}
            </Button>
          </CardFooter>
        </form>
      </Tabs>

      {generatedImage && (
        <CardContent>
          <h3 className="text-lg font-semibold mb-4">Your Generated Art:</h3>
          <img src={generatedImage} alt="Generated art" className="rounded-lg w-full" />
          <Button asChild variant="outline" className="w-full mt-4">
            <a href={generatedImage} download="generated-art.png">Download Image</a>
          </Button>
        </CardContent>
      )}
    </Card>
  )
}

// --- 3. Gallery Page ---
function GalleryPage() {
  const [images, setImages] = useState<GalleryImage[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchGallery = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/gallery/`)
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || "Failed to load gallery.")
      }
      const data: GalleryImage[] = await response.json()
      setImages(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch images when component mounts
  useEffect(() => {
    fetchGallery()
  }, [])

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Image Gallery</CardTitle>
          <CardDescription>All your previously generated images.</CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={fetchGallery} disabled={isLoading}>
          {isLoading ? <Spinner className="mr-2 h-4 w-4" /> : null}
          Refresh
        </Button>
      </CardHeader>
      <CardContent>
        {isLoading && <div className="flex justify-center p-12"><Spinner className="h-8 w-8" /></div>}
        {error && <p className="text-destructive text-sm text-center">{error}</p>}

        {!isLoading && !error && images.length === 0 && (
          <p className="text-muted-foreground text-center p-12">
            Your gallery is empty. Go to the "Generate" tab to create some art!
          </p>
        )}

        {!isLoading && !error && images.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {images.map((image) => (
              <Dialog key={image.filename}>
                <DialogTrigger asChild>
                  <button className="overflow-hidden rounded-lg group">
                    <img
                      src={`${API_BASE_URL}${image.url}`}
                      alt={image.filename}
                      className="aspect-square w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                    />
                  </button>
                </DialogTrigger>
                <DialogContent className="max-w-3xl">
                  <DialogHeader>
                    <DialogTitle>{image.filename.replace(/_/g, " ").replace(".png", "")}</DialogTitle>
                    <DialogDescription>
                      <a href={`${API_BASE_URL}${image.url}`} download={image.filename} className="text-sm text-blue-500 hover:underline">
                        Download full image
                      </a>
                    </DialogDescription>
                  </DialogHeader>
                  <img src={`${API_BASE_URL}${image.url}`} alt={image.filename} className="rounded-lg w-full" />
                </DialogContent>
              </Dialog>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default App