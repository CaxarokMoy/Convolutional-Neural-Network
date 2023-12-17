# Розробник Петренко Іван

Ця нейронна мережа призначена для класифікації зображень рукописних цифр з набору даних MNIST. Вона має архітектуру з двома згортковими шарами, двома повнозв'язаними шарами та деякою регуляризацією у вигляді dropout.

RNN_EPOTH.py

Коротко процес тренування виглядає наступним чином:

    1. Завантаження набору даних MNIST та створення DataLoader для тренування та тестування.
    2. Створення моделі з двома згортковими шарами та двома повнозв'язаними шарами.
    3. Використання AdamW як оптимізатора та кросс-ентропійної функції втрат для навчання.
    4. Використання шедулера кроку для динамічного зменшення швидкості навчання.
    5. Тренування моделі протягом кількох епох (15 епох у даному коді).
    6. Виведення прогресу тренування та оцінка точності на тестовому наборі під час тренування.
    7. Збереження моделі, оптимізатора та інших параметрів при кращій точності на тестовому наборі під час тренування.
    8. По завершенні всіх епох збереження остаточного стану моделі та оптимізатора.

Таким чином, нейронна мережа тренується для класифікації цифр у наборі даних MNIST, і тренується протягом 15 епох.

# Developer Petrenko Ivan 

This neural network is designed for the classification of handwritten digit images from the MNIST dataset. It has an architecture with two convolutional layers, two fully connected layers, and some regularization in the form of dropout.

RNN_EPOTH.py

Briefly, the training process looks as follows:

    1. Load the MNIST dataset and create DataLoaders for training and testing.
    2. Create a model with two convolutional layers and two fully connected layers.
    3. Use AdamW as the optimizer and cross-entropy loss function for training.
    4. Utilize a step scheduler for dynamically adjusting the learning rate.
    5. Train the model for a few epochs (15 epochs in this code).
    6. Display training progress and evaluate accuracy on the test dataset during training.
    7. Save the model, optimizer, and other parameters when achieving the best accuracy on the test dataset during training.
    8. After completing all epochs, save the final state of the model and optimizer.

In summary, the neural network is trained for the classification of digits in the MNIST dataset and undergoes training for 15 epochs.
