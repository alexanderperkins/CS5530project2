Name: Alex Perkins

Group Members: none

Video Links: N/A

Operating System: Windows 10 Pro
IDE: Visual Studio Code with Visual Studio Professional 2022 compiler

Instructions for Running:
Navigate to project2/src/build/Release/ and run
.\imgSearch.exe <target image path> <database dir> <# of top matches> <method>
Examples shown below after methods.

Methods:
baseline      - 7x7 center square with sum-of-squared-differences distance
histogram     - RGB color histogram with intersection distance
multihistogram - whole image and center histogram with weighted distance
texture       - color and texture histogram with equal weighted distance
dnn           - ResNet18 embeddings with SSD distance
custom        - horizontal edge density per row with SSD distance

Examples:
Task1
    .\imgSearch.exe C:\Users\Owner\source\repos\project2\data\pic.1016.jpg C:\Users\Owner\source\repos\project2\data 3 baseline
Task2
    .\imgSearch.exe C:\Users\Owner\source\repos\project2\data\pic.0164.jpg C:\Users\Owner\source\repos\project2\data 3 histogram
Task3
    .\imgSearch.exe C:\Users\Owner\source\repos\project2\data\pic.0274.jpg C:\Users\Owner\source\repos\project2\data 3 multihistogram
Task4
    .\imgSearch.exe C:\Users\Owner\source\repos\project2\data\pic.0535.jpg C:\Users\Owner\source\repos\project2\data 3 texture
Task5
    .\imgSearch.exe pic.0893.jpg C:\Users\Owner\source\repos\project2\data 3 dnn
    .\imgSearch.exe pic.0164.jpg C:\Users\Owner\source\repos\project2\data 3 dnn
Task6
    .\imgSearch.exe C:\Users\Owner\source\repos\project2\data\pic.1072.jpg C:\Users\Owner\source\repos\project2\data 5 histogram
    .\imgSearch.exe pic.1072.jpg C:\Users\Owner\source\repos\project2\data 5 dnn
    .\imgSearch.exe C:\Users\Owner\source\repos\project2\data\pic.0948.jpg C:\Users\Owner\source\repos\project2\data 5 histogram
    .\imgSearch.exe pic.0948.jpg C:\Users\Owner\source\repos\project2\data 5 dnn
    .\imgSearch.exe C:\Users\Owner\source\repos\project2\data\pic.0734.jpg C:\Users\Owner\source\repos\project2\data 5 histogram
    .\imgSearch.exe pic.0734.jpg C:\Users\Owner\source\repos\project2\data 5 dnn
Task7
    .\imgSearch.exe C:\Users\Owner\source\repos\project2\data\pic.0104.jpg C:\Users\Owner\source\repos\project2\data 5 custom

Extensions: GUI added to pop up 2 windows: 1 for target image, 2nd for top N image matches

Time Travel Days: 1