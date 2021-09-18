# LR_Scheduling

A repository to store the code required to implement the TCP "congestion congestion avoidance stategy" for branched models to acquire "knowledge" at equal rates

Data used:
- CIFAR 10

Model used:
- ResNet18

Congestion avoidance versions:
- "congestion_avoid" --> Implemented to be used on a ResNet18 model with two parallel branches. Each branch is learning to classify one class of images
- "congestion_avoid_10classes" --> Implemented to be used on a ResNet18 model learning to classify all ten image classes from CIFAR-10

Use "congestion_avoider_results.py" to produce the results from training the model with the inclusion of the congestion avoidance strategy. Implementations for the binary classification & multi-class classification tasks are included here
