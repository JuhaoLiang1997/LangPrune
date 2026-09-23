# Multilingual mC4 data configuration for Lang-Prune
# Supports C4 dataset and additional datasets for Aya-Expanse and Qwen3 models

c4_multilingual_data = {
    "ar": {
        "name": "arabic",
        "language_family": "Afro-Asian",
        "train": "multilingual/c4-ar.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-ar-validation.tfrecord-00000-of-00004.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "iw": {
        "name": 'hebrew',
        "language_family": "Afro-Asian",
        "train": "multilingual/c4-iw.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-iw-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "cs": {
        "name": "czech",
        "language_family": "Indo-European/Slavic",
        "train": "multilingual/c4-cs.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-cs-validation.tfrecord-00000-of-00002.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "ru": {
        "name": "russian",
        "language_family": "Indo-European/Slavic",
        "train": "multilingual/c4-ru.tfrecord-00000-of-04096.json.gz",
        "validation": "multilingual/c4-ru-validation.tfrecord-00000-of-00032.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "de": {
        "name": "german",
        "language_family": "Indo-European/Germanic",
        "train": "multilingual/c4-de.tfrecord-00000-of-02048.json.gz",
        "validation": "multilingual/c4-de-validation.tfrecord-00000-of-00016.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "en": {
        "name": "english",
        "language_family": "Indo-European/Germanic",
        "train": "en/c4-train.00000-of-01024.json.gz",
        "validation": "en/c4-validation.00000-of-00008.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "es": {
        "name": "spanish",
        "language_family": "Indo-European/Romance",
        "train": "multilingual/c4-es.tfrecord-00000-of-02048.json.gz",
        "validation": "multilingual/c4-es-validation.tfrecord-00000-of-00016.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "id": {
        "name": "indonesian",
        "language_family": "Austronesian",
        "train": "multilingual/c4-id.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-id-validation.tfrecord-00000-of-00004.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "zh": {
        "name": "chinese",
        "language_family": "Sino-Tibetan",
        "train": "multilingual/c4-zh.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-zh-validation.tfrecord-00000-of-00002.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    # Additional languages for Aya-Expanse and Qwen3
    "fr": {
        "name": "french",
        "language_family": "Indo-European/Romance",
        "train": "multilingual/c4-fr.tfrecord-00000-of-02048.json.gz",
        "validation": "multilingual/c4-fr-validation.tfrecord-00000-of-00016.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "it": {
        "name": "italian",
        "language_family": "Indo-European/Romance",
        "train": "multilingual/c4-it.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-it-validation.tfrecord-00000-of-00008.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "pt": {
        "name": "portuguese",
        "language_family": "Indo-European/Romance",
        "train": "multilingual/c4-pt.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-pt-validation.tfrecord-00000-of-00004.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "nl": {
        "name": "dutch",
        "language_family": "Indo-European/Germanic",
        "train": "multilingual/c4-nl.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-nl-validation.tfrecord-00000-of-00004.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "pl": {
        "name": "polish",
        "language_family": "Indo-European/Slavic",
        "train": "multilingual/c4-pl.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-pl-validation.tfrecord-00000-of-00004.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "tr": {
        "name": "turkish",
        "language_family": "Altaic",
        "train": "multilingual/c4-tr.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-tr-validation.tfrecord-00000-of-00004.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "ja": {
        "name": "japanese",
        "language_family": "Japonic",
        "train": "multilingual/c4-ja.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-ja-validation.tfrecord-00000-of-00008.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "ko": {
        "name": "korean",
        "language_family": "Koreanic",
        "train": "multilingual/c4-ko.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-ko-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "hi": {
        "name": "hindi",
        "language_family": "Indo-European/Indo-Aryan",
        "train": "multilingual/c4-hi.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-hi-validation.tfrecord-00000-of-00002.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "bn": {
        "name": "bengali",
        "language_family": "Indo-European/Indo-Aryan",
        "train": "multilingual/c4-bn.tfrecord-00000-of-00512.json.gz",
        "validation": "multilingual/c4-bn-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "th": {
        "name": "thai",
        "language_family": "Tai-Kadai",
        "train": "multilingual/c4-th.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-th-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "vi": {
        "name": "vietnamese",
        "language_family": "Austroasiatic",
        "train": "multilingual/c4-vi.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-vi-validation.tfrecord-00000-of-00004.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "fa": {
        "name": "persian",
        "language_family": "Indo-European",
        "train": "multilingual/c4-fa.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-fa-validation.tfrecord-00000-of-00002.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "ur": {
        "name": "urdu",
        "language_family": "Indo-European",
        "train": "multilingual/c4-ur.tfrecord-00000-of-00128.json.gz",
        "validation": "multilingual/c4-ur-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": True
    },
    "am": {
        "name": "amharic",
        "language_family": "Afro-Asian",
        "train": "multilingual/c4-am.tfrecord-00000-of-00016.json.gz",
        "validation": "multilingual/c4-am-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": False,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": False
    },
    "bg": {
        "name": "bulgarian",
        "language_family": "Indo-European/Slavic",
        "train": "multilingual/c4-bg.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-bg-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": False,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": False
    },
    "uk": {
        "name": "ukrainian",
        "language_family": "Indo-European/Slavic",
        "train": "multilingual/c4-uk.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-uk-validation.tfrecord-00000-of-00002.json.gz",
        "is_globalmmlu_supported": False,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": False
    },
    "sv": {
        "name": "swedish",
        "language_family": "Indo-European/Germanic",
        "train": "multilingual/c4-sv.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-sv-validation.tfrecord-00000-of-00002.json.gz",
        "is_globalmmlu_supported": False,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": False
    },
    "da": {
        "name": "danish",
        "language_family": "Indo-European/Germanic",
        "train": "multilingual/c4-da.tfrecord-00000-of-01024.json.gz",
        "validation": "multilingual/c4-da-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": False,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": False
    }, 
    "ms": {
        "name": "malay",
        "language_family": "Austronesian",
        "train": "multilingual/c4-ms.tfrecord-00000-of-00512.json.gz",
        "validation": "multilingual/c4-ms-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": True,
        "is_ayaexpanse_supported": False,
        "is_qwen3_supported": True
    },
    "my": {
        "name": "burmese",
        "language_family": "Sino-Tibetan",
        "train": "multilingual/c4-my.tfrecord-00000-of-00064.json.gz",
        "validation": "multilingual/c4-my-validation.tfrecord-00000-of-00001.json.gz",
        "is_globalmmlu_supported": False,
        "is_ayaexpanse_supported": True,
        "is_qwen3_supported": False
    },
}

# Additional datasets for specific model families
aya_expanse_datasets = {
    "oscar": {
        "en": "oscar-corpus/OSCAR-2201",
        "es": "oscar-corpus/OSCAR-2201",
        "fr": "oscar-corpus/OSCAR-2201",
        "de": "oscar-corpus/OSCAR-2201",
        "it": "oscar-corpus/OSCAR-2201",
        "pt": "oscar-corpus/OSCAR-2201",
        "nl": "oscar-corpus/OSCAR-2201",
        "pl": "oscar-corpus/OSCAR-2201",
        "ru": "oscar-corpus/OSCAR-2201",
        "cs": "oscar-corpus/OSCAR-2201",
        "tr": "oscar-corpus/OSCAR-2201",
        "ar": "oscar-corpus/OSCAR-2201",
        "hi": "oscar-corpus/OSCAR-2201",
        "bn": "oscar-corpus/OSCAR-2201",
        "th": "oscar-corpus/OSCAR-2201",
        "vi": "oscar-corpus/OSCAR-2201",
        "ja": "oscar-corpus/OSCAR-2201",
        "ko": "oscar-corpus/OSCAR-2201",
        "zh": "oscar-corpus/OSCAR-2201",
        "id": "oscar-corpus/OSCAR-2201"
    }
}

qwen3_datasets = {
    "pile": {
        "en": "EleutherAI/pile",
        "multilingual": "EleutherAI/pile"
    },
    "wikipedia": {
        "en": "wikipedia",
        "multilingual": "wikipedia"
    }
}

def get_supported_languages(model_family: str = None) -> list:
    """
    Get list of supported languages for a specific model family.
    
    Args:
        model_family: Model family (aya-expanse, qwen, etc.)
        
    Returns:
        List of supported language codes
    """
    if model_family == "aya-expanse":
        return [lang for lang, config in c4_multilingual_data.items() 
                if config.get("is_ayaexpanse_supported", False)]
    elif model_family == "qwen":
        return [lang for lang, config in c4_multilingual_data.items() 
                if config.get("is_qwen3_supported", False)]
    else:
        return list(c4_multilingual_data.keys())

def get_dataset_for_language(language: str, model_family: str = None, dataset_type: str = "c4") -> dict:
    """
    Get dataset configuration for a specific language and model family.
    
    Args:
        language: Language code
        model_family: Model family
        dataset_type: Type of dataset (c4, oscar, pile, etc.)
        
    Returns:
        Dataset configuration dictionary
    """
    if dataset_type == "c4":
        if language in c4_multilingual_data:
            return c4_multilingual_data[language]
        else:
            raise ValueError(f"Language {language} not supported in C4 dataset")
    
    elif dataset_type == "oscar" and model_family == "aya-expanse":
        if language in aya_expanse_datasets["oscar"]:
            return {"dataset": aya_expanse_datasets["oscar"][language]}
        else:
            raise ValueError(f"Language {language} not supported in OSCAR dataset")
    
    elif dataset_type in ["pile", "wikipedia"] and model_family == "qwen":
        if language in qwen3_datasets[dataset_type]:
            return {"dataset": qwen3_datasets[dataset_type][language]}
        else:
            raise ValueError(f"Language {language} not supported in {dataset_type} dataset")
    
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported for model family {model_family}")

def get_all_supported_languages() -> dict:
    """
    Get all supported languages by model family.
    
    Returns:
        Dictionary mapping model families to supported languages
    """
    return {
        "aya-expanse": get_supported_languages("aya-expanse"),
        "qwen": get_supported_languages("qwen"),
        "llama": get_supported_languages("llama"),
        "all": get_supported_languages()
    }