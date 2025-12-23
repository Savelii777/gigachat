# AI Агент на GigaChat и Salute Speech

AI-агент для автоматизации звонков Исполнителям с использованием GigaChat, Salute Speech, Voximplant и Qdrant.

## Описание

Этот проект реализует AI-агента, который:

1. **Интегрируется с АТС (Voximplant)** для совершения звонков Исполнителям заказов
2. **Ведёт диалог с Исполнителем** используя связку:
   - **Salute Speech** - распознавание речи (STT) и синтез речи (TTS)
   - **GigaChat** - генерация ответов и управление диалогом
   - **Qdrant** - векторная база данных с базой знаний компании (RAG)

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                        AI Agent                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Voximplant  │  │ Salute Speech│  │   GigaChat   │       │
│  │  (АТС/Звонки)│  │  (STT/TTS)   │  │  (Диалог)    │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └────────────────┬┴─────────────────┘                │
│                          │                                   │
│                  ┌───────┴───────┐                          │
│                  │    Qdrant     │                          │
│                  │ (База знаний) │                          │
│                  └───────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Установка

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/Savelii777/gigachat.git
cd gigachat
```

### 2. Создайте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate  # Windows
```

### 3. Установите зависимости

```bash
pip install -r requirements.txt
```

### 4. Настройте переменные окружения

```bash
cp .env.example .env
# Отредактируйте .env файл и добавьте свои API ключи
```

## Конфигурация

### Получение API ключей

#### GigaChat
1. Зарегистрируйтесь на [developers.sber.ru](https://developers.sber.ru)
2. Создайте проект GigaChat API
3. Получите Authorization Key в настройках API

#### Salute Speech
1. Зарегистрируйтесь на [developers.sber.ru](https://developers.sber.ru)
2. Подключите сервис SaluteSpeech
3. Получите API ключ

#### Voximplant
1. Зарегистрируйтесь на [voximplant.com](https://voximplant.com)
2. Создайте приложение и сценарий для звонков
3. Скачайте файл с credentials (JSON)

#### Qdrant
1. Установите Qdrant локально или используйте облачную версию
2. Для локальной установки: `docker run -p 6333:6333 qdrant/qdrant`

### Переменные окружения

| Переменная | Описание |
|------------|----------|
| `GIGACHAT_CREDENTIALS` | Ключ авторизации GigaChat |
| `GIGACHAT_SCOPE` | Область API (`GIGACHAT_API_PERS` или `GIGACHAT_API_B2B`) |
| `SALUTE_SPEECH_CREDENTIALS` | API ключ Salute Speech |
| `VOXIMPLANT_CREDENTIALS_PATH` | Путь к файлу credentials Voximplant |
| `VOXIMPLANT_RULE_ID` | ID правила/сценария в Voximplant |
| `QDRANT_URL` | URL сервера Qdrant |
| `AGENT_NAME` | Имя AI-агента |
| `COMPANY_NAME` | Название компании |

## Использование

### Базовый пример

```python
import asyncio
from src.utils.config import Config
from src.ai_agent import AIAgent
from src.integrations.voximplant_client import ExecutorInfo
from src.ai_agent.agent import OrderInfo

async def main():
    # Загрузка конфигурации
    config = Config.from_env()
    
    # Создание агента
    agent = AIAgent(config)
    
    # Добавление документов в базу знаний
    agent.add_knowledge_documents([
        {"content": "Информация о компании и услугах..."},
        {"content": "Правила работы с заказами..."},
    ])
    
    # Определение исполнителя
    executor = ExecutorInfo(
        executor_id="exec_001",
        name="Иван Петров",
        phone_number="+79991234567",
        skills=["delivery"],
        is_available=True,
    )
    
    # Определение заказа
    order = OrderInfo(
        order_id="ORD-001",
        description="Доставка мебели",
        address="ул. Пушкина, д. 10",
        datetime="Сегодня, 15:00",
        payment="3500 рублей",
    )
    
    # Совершение звонка
    session_id = await agent.call_executor(executor, order)
    
    if session_id:
        # Генерация приветствия
        greeting, audio = await agent.generate_initial_greeting(session_id)
        print(f"Агент: {greeting}")
    
    agent.close()

asyncio.run(main())
```

### Запуск примера

```bash
python examples/basic_usage.py
```

## Структура проекта

```
gigachat/
├── src/
│   ├── ai_agent/
│   │   ├── __init__.py
│   │   └── agent.py           # Основной класс AI Agent
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── gigachat_client.py    # Клиент GigaChat
│   │   ├── salute_speech_client.py  # Клиент Salute Speech
│   │   ├── voximplant_client.py  # Клиент Voximplant
│   │   └── qdrant_client.py      # Клиент Qdrant
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Конфигурация
├── examples/
│   ├── __init__.py
│   └── basic_usage.py         # Пример использования
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Компоненты

### AIAgent
Основной класс, который координирует работу всех компонентов:
- Инициирует звонки через Voximplant
- Управляет диалогом с помощью GigaChat
- Использует Salute Speech для STT/TTS
- Извлекает контекст из базы знаний Qdrant

### GigaChatClient
Клиент для работы с GigaChat API:
- Генерация ответов на основе контекста
- Анализ намерений пользователя
- Создание системных промптов

### SaluteSpeechClient
Клиент для работы с Salute Speech:
- Распознавание речи (Speech-to-Text)
- Синтез речи (Text-to-Speech)

### VoximplantClient
Клиент для работы с Voximplant API:
- Инициирование звонков
- Управление сценариями
- Отправка SMS

### QdrantKnowledgeBase
Клиент для работы с векторной базой данных:
- Добавление документов
- Поиск релевантной информации
- RAG (Retrieval-Augmented Generation)

## Разработка

### Требования
- Python 3.10+
- Доступ к API GigaChat, Salute Speech, Voximplant
- Qdrant (локально или в облаке)

### Тестирование
```bash
# Установка dev зависимостей
pip install pytest pytest-asyncio

# Запуск тестов
pytest
```

## Лицензия

MIT License

## Авторы

- [Savelii777](https://github.com/Savelii777)
