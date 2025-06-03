import cv2
import numpy as np

# Global variables to store depth data
current_depth_image = None
window_name = ""

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to display depth values on hover"""
    global current_depth_image, window_name
    
    if current_depth_image is not None:
        h, w = current_depth_image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            # Get depth value at mouse position
            depth_value = current_depth_image[y, x]
            
            # Create a copy of the displayed image to draw on
            display_img = param.copy()
            
            # Draw crosshair at mouse position
            cv2.line(display_img, (x-10, y), (x+10, y), (0, 255, 0), 1)
            cv2.line(display_img, (x, y-10), (x, y+10), (0, 255, 0), 1)
            
            # Display depth value as text
            text = f"Depth: {depth_value:.3f}"
            cv2.putText(display_img, text, (x+15, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_img, f"({x},{y})", (x+15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Update the display
            cv2.imshow(window_name, display_img)

def read_and_display_depth_interactive(filename, method="16bit"):
    """Interactive depth image viewer with hover functionality"""
    global current_depth_image, window_name
    
    # Read depth image
    depth_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"Error: Could not load image {filename}")
        return None
    
    current_depth_image = depth_image
    
    print(f"Depth image shape: {depth_image.shape}")
    print(f"Depth image dtype: {depth_image.dtype}")
    print(f"Min depth: {np.min(depth_image)}, Max depth: {np.max(depth_image)}")
    print("Hover over the image to see depth values. Press 'q' or ESC to exit.")
    
    # Process based on method
    if method == "32bit":
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)
        depth_display = cv2.convertScaleAbs(depth_image, alpha=255.0/np.max(depth_image))
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_VIRIDIS)
        window_name = "Interactive 32-bit Depth Image"
        
    elif method == "enhanced":
        # Enhanced visualization
        valid_mask = (depth_image > 0) & (depth_image < 65535)
        valid_depths = depth_image[valid_mask]
        
        if len(valid_depths) == 0:
            print("No valid depth values found")
            return None
        
        min_depth = np.percentile(valid_depths, 1)
        max_depth = np.percentile(valid_depths, 99)
        depth_clipped = np.clip(depth_image, min_depth, max_depth)
        depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        window_name = "Interactive Enhanced Depth Image"
        
    else:  # 16bit default
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = depth_normalized.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        window_name = "Interactive 16-bit Depth Image"
    
    # Create window and set mouse callback
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, depth_colored)
    
    # Display the image
    cv2.imshow(window_name, depth_colored)
    
    # Wait for key press
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC key
            break
    
    cv2.destroyAllWindows()
    return current_depth_image

def read_and_display_depth_with_click(filename):
    """Alternative version that shows depth on mouse click"""
    global current_depth_image, window_name
    
    def click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_depth_image is not None:
                h, w = current_depth_image.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    depth_value = current_depth_image[y, x]
                    print(f"Clicked at ({x}, {y}): Depth = {depth_value:.3f}")
    
    depth_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"Error: Could not load image {filename}")
        return None
    
    current_depth_image = depth_image
    window_name = "Click for Depth Values"
    
    # Normalize and colorize
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_callback)
    cv2.imshow(window_name, depth_colored)
    
    print("Click on the image to see depth values. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return current_depth_image

# Usage examples:
if __name__ == "__main__":
    # Replace with your depth image filename
    depth_filename = "depth_16bit.png"
    
    try:
        # Interactive hover version (recommended)
        depth = read_and_display_depth_interactive(depth_filename, method="16bit")
        
        # Alternative methods:
        # depth = read_and_display_depth_interactive(depth_filename, method="32bit")
        # depth = read_and_display_depth_interactive(depth_filename, method="enhanced")
        
        # Click version (alternative)
        # depth = read_and_display_depth_with_click(depth_filename)
        
    except Exception as e:
        print(f"Error: {e}")
