# -SCHOOL--Computer-Graphics-Vulkan
This project's idea was to:
- write a Vulkan program that displays (rasterizes) a coloured 2D triangle on the screen.
  - Make the triangle move in 2D
- Add a texture onto the triangle instead of the per-vertex colouring.
- Load and render a custom 3D model from a file.

# high-level description of what the program does in order to render the triangle

- First a window is created as well as a Vulkan instance which is required for using Vulkan.
- A surface is created which can be used for rendering onto and it is bound to the window. 
- A logical device is created which is an interface to the physical device (GPU).  
- Next the render pipeline is created. 
- The first part is the swapchain, which is a queue of one or more images and their corresponding image views which contain rendered frames. 
- The renderer places frames at the back of the queue, while the render surface takes frames from the front to present. 
- A renderpass object defines more specific information about how an image is rendered, such as what attachments it contains (for example color/depth), how they should be used and so on. 
- The pipeline object then wraps all this into a neat package, including other information such as shader data.
- Shaders define how incoming non-visual data is displayed in a visual format. For example a shader could be given vertex coordinates and colors in vector format, and output pixel color values.
- Framebuffers are created for each part of the swapchain to combine all the different image parts
that make up a frame, as defined by the renderpass.
- Next, mesh information is uploaded to device memory, and the GPU is given instructions of what is required to create a frame. 
- These are done using recorded commands which the GPU then later executes once (in the case of moving data to device memory) or every time it is asked to render a frame.  
- Finally, the GPU is asked to draw frames, and it does so as per all the previous instructions.
