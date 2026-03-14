# ДЗ 2: Классификация FashionMNIST с PyTorch Lightning

Классификация изображений одежды (10 классов) с использованием PyTorch Lightning.

## Задание

Реализовать модель для классификации изображений датасета FashionMNIST:

1. **FashionMNISTDataModule (3 балла)** - загрузка, предобработка, split, dataloaders
2. **FashionMNISTModel (3 балла)** - CNN модель с метриками F1, ROC AUC
3. **Обучение с Trainer (2 балла)** - EarlyStopping, TensorBoard, визуализация
4. **Обоснование решений (1 балл)** - выбор архитектуры и гиперпараметров
5. **Воспроизводимость (1 балл)** - фиксация seed, детерминированность

## Структура

```
hw_2/
├── FashionMNIST_Lightning.ipynb  # Основной ноутбук
├── README.md                      # Этот файл
└── .gitignore                     # Игнорируемые файлы
```

## Использование

1. Установите зависимости:
```bash
pip install torch torchvision pytorch-lightning torchmetrics tensorboard pandas matplotlib seaborn scikit-learn
```

2. Запустите ноутбук:
```bash
jupyter notebook FashionMNIST_Lightning.ipynb
```

3. Для визуализации TensorBoard:
```bash
tensorboard --logdir=lightning_logs
```

## Архитектура модели

**CNN:**
- Conv1: 1→32 (3x3) + BatchNorm + ReLU + MaxPool
- Conv2: 32→64 (3x3) + BatchNorm + ReLU + MaxPool
- FC1: 3136→128 + Dropout(0.3)
- FC2: 128→10

**Параметров:** ~130k

## Гиперпараметры

- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Batch size:** 128
- **Max epochs:** 30
- **EarlyStopping:** patience=5

## Ожидаемые результаты

- **Test Accuracy:** 90-92%
- **Test F1:** 0.89-0.91
- **Test AUROC:** 0.98-0.99

## Воспроизводимость

- Random seed: 42
- Deterministic mode: True
- Все зависимости фиксированы

## Автор

Домашнее задание по курсу Advanced ML/DL, ITMO

