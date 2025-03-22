#include "raylib.h"
#include <stdio.h>

int main() {
    InitWindow(1, 1, "rendering image...");
    SetWindowState(FLAG_WINDOW_HIDDEN);

    if (!IsWindowReady()) {
        printf("Failed to initialize window\n");
        return 1;
    }

    RenderTexture2D target = LoadRenderTexture(800, 600);
    if (target.id == 0) {
        printf("Failed to create render texture\n");
        return 1;
    }
    
    BeginTextureMode(target);
        ClearBackground(RAYWHITE);
        DrawCircle(400, 300, 100, RED);
        DrawText("Rendered offscreen!", 200, 250, 30, BLACK);
    EndTextureMode();
    
    Image image = LoadImageFromTexture(target.texture);
    
    if (image.data == NULL) {
        printf("Failed to get image data from texture\n");
        UnloadRenderTexture(target);
        return 1;
    }
    
    ImageFlipVertical(&image);
    
    const char* outputPath = "./output_image.png";
    
    bool success = ExportImage(image, outputPath);
    if (!success) {
        printf("Failed to export image to: %s\n", outputPath);
    } else {
        printf("Image successfully saved to: %s\n", outputPath);
    }

    UnloadImage(image);
    UnloadRenderTexture(target);
    CloseWindow();
    
    return 0;
}

