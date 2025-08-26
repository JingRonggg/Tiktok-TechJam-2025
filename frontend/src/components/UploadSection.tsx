import { useState, useCallback } from "react";
import {
  Upload,
  File,
  Image,
  Video,
  X,
  CheckCircle2,
  Clock,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

interface FileItem {
  id: string;
  file: File;
  status: "uploading" | "processing" | "completed" | "error";
  progress: number;
  processedUrl?: string;
}

const UploadSection = () => {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const { toast } = useToast();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        const selectedFiles = Array.from(e.target.files);
        handleFiles(selectedFiles);
      }
    },
    []
  );

  const handleFiles = (newFiles: File[]) => {
    const validFiles = newFiles.filter((file) => {
      const isValid =
        file.type.startsWith("image/") || file.type.startsWith("video/");
      if (!isValid) {
        toast({
          title: "Invalid file type",
          description: `${file.name} is not a supported image or video file.`,
          variant: "destructive",
        });
      }
      return isValid;
    });

    const fileItems: FileItem[] = validFiles.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: "uploading",
      progress: 0,
    }));

    setFiles((prev) => [...prev, ...fileItems]);

    // Simulate upload and processing
    fileItems.forEach((fileItem) => {
      simulateProcessing(fileItem.id);
    });
  };

  const simulateProcessing = (fileId: string) => {
    // Simulate upload progress
    let progress = 0;
    const uploadInterval = setInterval(() => {
      progress += Math.random() * 20;
      if (progress >= 100) {
        progress = 100;
        clearInterval(uploadInterval);

        setFiles((prev) =>
          prev.map((f) =>
            f.id === fileId ? { ...f, status: "processing", progress: 0 } : f
          )
        );

        // Simulate AI processing
        setTimeout(() => {
          let processProgress = 0;
          const processInterval = setInterval(() => {
            processProgress += Math.random() * 15;
            if (processProgress >= 100) {
              processProgress = 100;
              clearInterval(processInterval);

              setFiles((prev) =>
                prev.map((f) =>
                  f.id === fileId
                    ? {
                        ...f,
                        status: "completed",
                        progress: 100,
                        processedUrl: URL.createObjectURL(f.file), // Placeholder
                      }
                    : f
                )
              );

              toast({
                title: "Processing completed",
                description:
                  "Your file has been successfully protected and is ready for download.",
              });
            } else {
              setFiles((prev) =>
                prev.map((f) =>
                  f.id === fileId ? { ...f, progress: processProgress } : f
                )
              );
            }
          }, 300);
        }, 1000);
      } else {
        setFiles((prev) =>
          prev.map((f) => (f.id === fileId ? { ...f, progress } : f))
        );
      }
    }, 200);
  };

  const removeFile = (fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  const downloadFile = (fileId: string) => {
    const fileItem = files.find((f) => f.id === fileId);
    if (fileItem?.processedUrl) {
      // In a real app, this would download the processed file
      toast({
        title: "Download started",
        description: "Your protected file is being downloaded.",
      });
    }
  };

  return (
    <section id="upload" className="py-20 bg-background">
      <div className="container">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            Upload & Protect Your Files
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Drag and drop your images or videos below. Our AI will automatically
            detect and protect sensitive content.
          </p>
        </div>

        <div className="max-w-4xl mx-auto space-y-8">
          {/* Upload Area */}
          <Card
            className={`relative bg-gradient-card border-2 border-dashed transition-all duration-300 ${
              isDragOver
                ? "border-trust-blue bg-trust-blue/10 shadow-glow"
                : "border-border hover:border-trust-blue/50"
            }`}
          >
            <div
              className="p-12 text-center cursor-pointer"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById("file-input")?.click()}
            >
              <Upload
                className={`mx-auto h-16 w-16 mb-4 transition-colors ${
                  isDragOver ? "text-trust-blue" : "text-muted-foreground"
                }`}
              />
              <h3 className="text-xl font-semibold text-foreground mb-2">
                Drop files here or click to upload
              </h3>
              <p className="text-muted-foreground mb-6">
                Support for images (JPG, PNG, GIF) and videos (MP4, MOV, AVI)
              </p>
              <Button variant="upload" size="lg">
                <Upload className="mr-2 h-5 w-5" />
                Choose Files
              </Button>
              <input
                id="file-input"
                type="file"
                multiple
                accept="image/*,video/*"
                onChange={handleFileInput}
                className="hidden"
              />
            </div>
          </Card>

          {/* File List */}
          {files.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-foreground">
                Processing Queue
              </h3>
              {files.map((fileItem) => (
                <Card key={fileItem.id} className="bg-gradient-card">
                  <div className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        {fileItem.file.type.startsWith("image/") ? (
                          <Image className="h-8 w-8 text-trust-blue" />
                        ) : (
                          <Video className="h-8 w-8 text-trust-blue" />
                        )}
                        <div>
                          <p className="font-medium text-foreground">
                            {fileItem.file.name}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {(fileItem.file.size / (1024 * 1024)).toFixed(2)} MB
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        {fileItem.status === "completed" && (
                          <Button
                            variant="secure"
                            size="sm"
                            onClick={() => downloadFile(fileItem.id)}
                          >
                            Download
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => removeFile(fileItem.id)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">
                          {fileItem.status === "uploading" && "Uploading..."}
                          {fileItem.status === "processing" &&
                            "AI Processing..."}
                          {fileItem.status === "completed" &&
                            "Protected & Ready"}
                          {fileItem.status === "error" && "Error occurred"}
                        </span>
                        <span className="text-foreground">
                          {Math.round(fileItem.progress)}%
                        </span>
                      </div>

                      <div className="w-full bg-secondary rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-300 ${
                            fileItem.status === "uploading"
                              ? "bg-trust-blue"
                              : fileItem.status === "processing"
                              ? "bg-warning-amber"
                              : fileItem.status === "completed"
                              ? "bg-secure-green"
                              : "bg-destructive"
                          }`}
                          style={{ width: `${fileItem.progress}%` }}
                        />
                      </div>
                    </div>

                    {fileItem.status === "completed" && (
                      <div className="mt-4 flex items-center gap-2 text-secure-green">
                        <CheckCircle2 className="h-4 w-4" />
                        <span className="text-sm font-medium">
                          File protected and ready for download
                        </span>
                      </div>
                    )}

                    {fileItem.status === "processing" && (
                      <div className="mt-4 flex items-center gap-2 text-warning-amber">
                        <Clock className="h-4 w-4 animate-spin" />
                        <span className="text-sm font-medium">
                          AI is analyzing and protecting your content
                        </span>
                      </div>
                    )}
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default UploadSection;
