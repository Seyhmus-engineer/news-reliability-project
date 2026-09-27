# ============================================================
# TURKISH ELECTRA MODEL SERVİSİ
#
# Dosya:
# backend/app/services/model_service.py
#
# ============================================================

import logging
import re
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


logger = logging.getLogger(__name__)


class ModelService:
    """
    Turkish ELECTRA haber sınıflandırma modelini yöneten servis.

    Sınıflar:
        0 -> real
        1 -> fake
    """

    # Modelin özel tokenlar dâhil maksimum giriş uzunluğu.
    MAX_LENGTH = 512

    # Çok kısa içeriklerden güvenilir olmayan tahmin alınmasını
    # engelleyen minimum token sayısı.
    MIN_TOKEN_COUNT = 80

    # Uzun haber parçaları arasındaki token örtüşmesi.
    CHUNK_OVERLAP_TOKENS = 50

    # API'ye gönderilebilecek maksimum metin uzunluğu.
    MAX_CHARACTER_COUNT = 100_000

    def __init__(self) -> None:
        """
        Bu dosyanın konumu:

        project/
            backend/
                app/
                    services/
                        model_service.py

        parents[3] proje köküne karşılık gelir.
        """

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        self.model_dir = (
            self.project_root
            / "models"
            / "electra_turkish_article_60k_final"
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer: Any | None = None
        self.model: Any | None = None

        self._loaded = False

        # Modelin aynı anda birden fazla kez yüklenmesini önler.
        self._load_lock = Lock()

        # Aynı model üzerinde eş zamanlı tahmin yapılmasını önler.
        self._inference_lock = Lock()

    # ========================================================
    # MODEL DURUMU
    # ========================================================

    @property
    def is_loaded(self) -> bool:
        return (
            self._loaded
            and self.tokenizer is not None
            and self.model is not None
        )

    # ========================================================
    # MODEL KLASÖRÜ KONTROLÜ
    # ========================================================

    def _validate_model_directory(self) -> None:
        """
        Model klasörünün ve temel dosyaların mevcut olduğunu
        doğrular.
        """

        if not self.model_dir.exists():
            raise FileNotFoundError(
                "Model klasörü bulunamadı:\n"
                f"{self.model_dir}"
            )

        if not self.model_dir.is_dir():
            raise NotADirectoryError(
                "Model yolu bir klasör değil:\n"
                f"{self.model_dir}"
            )

        config_path = (
            self.model_dir
            / "config.json"
        )

        if not config_path.is_file():
            raise FileNotFoundError(
                "Model config dosyası bulunamadı:\n"
                f"{config_path}"
            )

        possible_weight_files = [
            self.model_dir / "model.safetensors",
            self.model_dir / "pytorch_model.bin",
        ]

        has_weight_file = any(
            path.is_file()
            for path in possible_weight_files
        )

        if not has_weight_file:
            raise FileNotFoundError(
                "Model ağırlık dosyası bulunamadı. "
                "Beklenen dosyalardan biri:\n"
                f"{possible_weight_files[0]}\n"
                f"{possible_weight_files[1]}"
            )

    # ========================================================
    # MODELİ YÜKLE
    # ========================================================

    def load_model(self) -> None:
        """
        Tokenizer ve modeli uygulama boyunca yalnızca bir kez
        yükler.
        """

        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return

            print("=" * 70)
            print("FINAL TURKISH ELECTRA MODELİ YÜKLENİYOR")
            print("=" * 70)

            print(
                "Proje kökü:",
                self.project_root,
            )

            print(
                "Model yolu:",
                self.model_dir,
            )

            print(
                "Kullanılan cihaz:",
                self.device,
            )

            if torch.cuda.is_available():
                print(
                    "GPU:",
                    torch.cuda.get_device_name(0),
                )

                # RTX GPU üzerinde uygun matris işlemlerinde
                # performans iyileştirmesine izin verir.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                torch.set_float32_matmul_precision(
                    "high"
                )

            try:
                self._validate_model_directory()

                self.tokenizer = (
                    AutoTokenizer.from_pretrained(
                        str(self.model_dir),
                        local_files_only=True,
                        use_fast=True,
                    )
                )

                self.model = (
                    AutoModelForSequenceClassification
                    .from_pretrained(
                        str(self.model_dir),
                        local_files_only=True,
                    )
                )

                self.model.to(
                    self.device
                )

                self.model.eval()

                number_of_labels = int(
                    self.model.config.num_labels
                )

                if number_of_labels != 2:
                    raise RuntimeError(
                        "Model iki sınıflı değil. "
                        "Beklenen num_labels: 2, "
                        f"bulunan: {number_of_labels}"
                    )

                self._validate_special_tokens()

                self._loaded = True

                print(
                    "Tokenizer sınıfı:",
                    self.tokenizer.__class__.__name__,
                )

                print(
                    "Model sınıfı:",
                    self.model.__class__.__name__,
                )

                print(
                    "Label eşlemesi:",
                    self.model.config.id2label,
                )

                print(
                    "CLS token ID:",
                    self.tokenizer.cls_token_id,
                )

                print(
                    "SEP token ID:",
                    self.tokenizer.sep_token_id,
                )

                print(
                    "Model başarıyla yüklendi."
                )

            except Exception:
                self._loaded = False
                self.tokenizer = None
                self.model = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.exception(
                    "Final Turkish ELECTRA modeli "
                    "yüklenemedi."
                )

                raise

    # ========================================================
    # ÖZEL TOKEN KONTROLÜ
    # ========================================================

    def _validate_special_tokens(self) -> None:
        """
        BERT/ELECTRA girişinde kullanılacak CLS ve SEP
        tokenlarının mevcut olduğunu doğrular.
        """

        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer kullanıma hazır değil."
            )

        cls_token_id = (
            self.tokenizer.cls_token_id
        )

        sep_token_id = (
            self.tokenizer.sep_token_id
        )

        if cls_token_id is None:
            raise RuntimeError(
                "Tokenizer CLS token ID içermiyor."
            )

        if sep_token_id is None:
            raise RuntimeError(
                "Tokenizer SEP token ID içermiyor."
            )

    # ========================================================
    # METİN TEMİZLEME
    # ========================================================

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Görünmeyen karakterleri, kontrol karakterlerini ve
        gereksiz boşlukları temizler.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Haber metni string olmalıdır."
            )

        cleaned_text = re.sub(
            r"[\u200B-\u200D\uFEFF]",
            "",
            text,
        )

        cleaned_text = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
            " ",
            cleaned_text,
        )

        cleaned_text = re.sub(
            r"\s+",
            " ",
            cleaned_text,
        ).strip()

        if not cleaned_text:
            raise ValueError(
                "Haber metni boş olamaz."
            )

        if len(cleaned_text) < 20:
            raise ValueError(
                "Haber metni en az 20 karakter olmalıdır."
            )

        if (
            len(cleaned_text)
            > ModelService.MAX_CHARACTER_COUNT
        ):
            raise ValueError(
                "Haber metni izin verilen karakter sınırını "
                "aşıyor. "
                f"Maksimum karakter sayısı: "
                f"{ModelService.MAX_CHARACTER_COUNT:,}"
            )

        return cleaned_text

    # ========================================================
    # LABEL EŞLEMESİ
    # ========================================================

    def _get_label_name(
        self,
        predicted_id: int,
    ) -> str:
        """
        Model label bilgisini real veya fake değerine çevirir.
        """

        if self.model is None:
            raise RuntimeError(
                "Model kullanıma hazır değil."
            )

        if predicted_id not in {0, 1}:
            raise RuntimeError(
                "Geçersiz tahmin sınıfı: "
                f"{predicted_id}"
            )

        id2label = (
            self.model.config.id2label
            or {}
        )

        predicted_label = id2label.get(
            predicted_id
        )

        if predicted_label is None:
            predicted_label = id2label.get(
                str(predicted_id)
            )

        normalized_label = str(
            predicted_label or ""
        ).strip().lower()

        known_label_mapping = {
            "real": "real",
            "fake": "fake",
            "label_0": "real",
            "label_1": "fake",
            "0": "real",
            "1": "fake",
        }

        mapped_label = known_label_mapping.get(
            normalized_label
        )

        if mapped_label is not None:
            return mapped_label

        # Model config dosyasında label adı bulunmadığında
        # proje sınıf sırası kullanılır.
        return (
            "real"
            if predicted_id == 0
            else "fake"
        )

    # ========================================================
    # METNİ TOKEN ID LİSTESİNE ÇEVİR
    # ========================================================

    def _tokenize_without_special_tokens(
        self,
        text: str,
    ) -> list[int]:
        """
        Metni CLS ve SEP eklemeden token ID listesine çevirir.

        Bu token listesi hem uzunluk kontrolünde hem de
        parçalama işleminde kullanılır.
        """

        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer kullanıma hazır değil."
            )

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        normalized_ids = [
            int(token_id)
            for token_id in token_ids
        ]

        if not normalized_ids:
            raise ValueError(
                "Haber metni tokenlara ayrılamadı."
            )

        return normalized_ids

    # ========================================================
    # MODELİN İÇERİK TOKEN KAPASİTESİ
    # ========================================================

    def _get_maximum_content_tokens(self) -> int:
        """
        Model girişinde bir CLS ve bir SEP tokenı kullanılır.

        512 - 2 = 510 içerik tokenı.
        """

        self._validate_special_tokens()

        special_token_count = 2

        maximum_content_tokens = (
            self.MAX_LENGTH
            - special_token_count
        )

        if maximum_content_tokens <= 0:
            raise RuntimeError(
                "Modelin içerik token kapasitesi geçersiz."
            )

        return maximum_content_tokens

    # ========================================================
    # TOKEN PARÇALARINI OLUŞTUR
    # ========================================================

    def _create_token_chunks(
        self,
        content_token_ids: list[int],
    ) -> list[dict[str, Any]]:
        """
        510 içerik tokenını aşan haberleri örtüşmeli parçalara
        ayırır.

        Kısa metinlerde yalnızca bir parça oluşturulur.
        """

        if not content_token_ids:
            raise ValueError(
                "Parçalanacak token listesi boş."
            )

        maximum_content_tokens = (
            self._get_maximum_content_tokens()
        )

        if (
            self.CHUNK_OVERLAP_TOKENS < 0
            or self.CHUNK_OVERLAP_TOKENS
            >= maximum_content_tokens
        ):
            raise RuntimeError(
                "Parça örtüşme token sayısı geçersiz."
            )

        total_content_tokens = len(
            content_token_ids
        )

        # 510 içerik tokenı veya daha kısa metin
        # kesinlikle bölünmez.
        if (
            total_content_tokens
            <= maximum_content_tokens
        ):
            return [
                {
                    "chunk_index": 1,
                    "start_index": 0,
                    "end_index": total_content_tokens,
                    "content_token_ids": (
                        content_token_ids
                    ),
                    "content_token_count": (
                        total_content_tokens
                    ),
                    "effective_token_count": (
                        total_content_tokens
                    ),
                }
            ]

        chunks: list[dict[str, Any]] = []

        start_index = 0
        previous_end_index = 0
        chunk_index = 1

        while start_index < total_content_tokens:
            end_index = min(
                start_index + maximum_content_tokens,
                total_content_tokens,
            )

            current_ids = content_token_ids[
                start_index:end_index
            ]

            if not current_ids:
                break

            if chunk_index == 1:
                effective_token_count = len(
                    current_ids
                )

            else:
                # Önceki parçada zaten bulunan örtüşme
                # tokenlarını ikinci kez ağırlıklandırmaz.
                effective_token_count = max(
                    1,
                    end_index - previous_end_index,
                )

            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_index": start_index,
                    "end_index": end_index,
                    "content_token_ids": current_ids,
                    "content_token_count": len(
                        current_ids
                    ),
                    "effective_token_count": (
                        effective_token_count
                    ),
                }
            )

            if end_index >= total_content_tokens:
                break

            previous_end_index = end_index

            start_index = (
                end_index
                - self.CHUNK_OVERLAP_TOKENS
            )

            chunk_index += 1

        if not chunks:
            raise RuntimeError(
                "Haber metni parçalara ayrılamadı."
            )

        return chunks

    # ========================================================
    # TOKEN PARÇASINI MODEL GİRDİSİNE ÇEVİR
    # ========================================================

    def _prepare_chunk_inputs(
        self,
        content_token_ids: list[int],
    ) -> dict[str, torch.Tensor]:

        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer kullanıma hazır değil."
            )

        if not content_token_ids:
            raise ValueError(
                "Analiz edilecek token parçası boş."
            )

        self._validate_special_tokens()

        cls_token_id = int(
            self.tokenizer.cls_token_id
        )

        sep_token_id = int(
            self.tokenizer.sep_token_id
        )

        normalized_content_ids = [
            int(token_id)
            for token_id in content_token_ids
        ]

        input_ids = [
            cls_token_id,
            *normalized_content_ids,
            sep_token_id,
        ]

        input_token_count = len(
            input_ids
        )

        if input_token_count > self.MAX_LENGTH:
            raise RuntimeError(
                "Hazırlanan model girdisi maksimum token "
                "sınırını aşıyor. "
                f"Bulunan: {input_token_count}, "
                f"maksimum: {self.MAX_LENGTH}"
            )

        attention_mask = [
            1
            for _ in input_ids
        ]

        model_inputs: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=self.device,
            ),

            "attention_mask": torch.tensor(
                [attention_mask],
                dtype=torch.long,
                device=self.device,
            ),
        }

        return model_inputs

    # ========================================================
    # TEK PARÇA TAHMİNİ
    # ========================================================

    def _predict_chunk(
        self,
        content_token_ids: list[int],
    ) -> dict[str, Any]:
        """
        Tek token parçası için real ve fake olasılıklarını üretir.
        """

        if self.model is None:
            raise RuntimeError(
                "Model kullanıma hazır değil."
            )

        model_inputs = self._prepare_chunk_inputs(
            content_token_ids
        )

        with torch.inference_mode():
            outputs = self.model(
                **model_inputs
            )

            logits = outputs.logits

        expected_shape = (
            1,
            2,
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Model tahmin çıktısı geçersiz. "
                f"Beklenen: {expected_shape}, "
                f"bulunan: {tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "Model geçersiz tahmin skorları üretti."
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0]

        real_probability = float(
            probabilities[0].item()
        )

        fake_probability = float(
            probabilities[1].item()
        )

        if not (
            0.0 <= real_probability <= 1.0
            and 0.0 <= fake_probability <= 1.0
        ):
            raise RuntimeError(
                "Model geçersiz olasılık değeri üretti."
            )

        predicted_id = int(
            torch.argmax(
                probabilities
            ).item()
        )

        predicted_label = self._get_label_name(
            predicted_id
        )

        confidence = max(
            real_probability,
            fake_probability,
        )

        input_token_count = int(
            model_inputs[
                "input_ids"
            ].shape[1]
        )

        return {
            "label": predicted_label,
            "label_id": predicted_id,
            "confidence": confidence,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "input_token_count": input_token_count,
        }

    # ========================================================
    # GENEL TAHMİN
    # ========================================================

    def predict(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        Haber metnini analiz eder.

        Akış:
            1. Metin temizlenir.
            2. Token sayısı hesaplanır.
            3. 80 tokendan kısa metin reddedilir.
            4. 512 token ve altı metin bölünmez.
            5. Daha uzun metin parçalanır.
            6. Parça olasılıkları ağırlıklı birleştirilir.
        """

        if not self.is_loaded:
            raise RuntimeError(
                "Model henüz yüklenmedi."
            )

        cleaned_text = self.clean_text(
            text
        )

        content_token_ids = (
            self._tokenize_without_special_tokens(
                cleaned_text
            )
        )

        # Orijinal haberin token sayısı:
        # içerik + CLS + SEP
        total_token_count = (
            len(content_token_ids)
            + 2
        )

        if total_token_count < self.MIN_TOKEN_COUNT:
            raise ValueError(
                "Metin güvenilir bir haber analizi için "
                "çok kısa. "
                f"En az {self.MIN_TOKEN_COUNT} token "
                "gereklidir; gönderilen metin "
                f"{total_token_count} token içeriyor. "
                "Lütfen haberin yalnızca başlığını değil, "
                "başlık, açıklama ve haber gövdesini gönderin."
            )

        chunks = self._create_token_chunks(
            content_token_ids
        )

        chunk_results: list[dict[str, Any]] = []

        weighted_real_sum = 0.0
        weighted_fake_sum = 0.0
        total_effective_weight = 0

        with self._inference_lock:
            for chunk in chunks:
                prediction = self._predict_chunk(
                    chunk[
                        "content_token_ids"
                    ]
                )

                effective_weight = int(
                    chunk[
                        "effective_token_count"
                    ]
                )

                weighted_real_sum += (
                    prediction[
                        "real_probability"
                    ]
                    * effective_weight
                )

                weighted_fake_sum += (
                    prediction[
                        "fake_probability"
                    ]
                    * effective_weight
                )

                total_effective_weight += (
                    effective_weight
                )

                chunk_results.append(
                    {
                        "chunk_index": int(
                            chunk[
                                "chunk_index"
                            ]
                        ),

                        "token_count": int(
                            prediction[
                                "input_token_count"
                            ]
                        ),

                        "effective_token_count": (
                            effective_weight
                        ),

                        "label": prediction[
                            "label"
                        ],

                        "label_id": int(
                            prediction[
                                "label_id"
                            ]
                        ),

                        "confidence": round(
                            float(
                                prediction[
                                    "confidence"
                                ]
                            ),
                            6,
                        ),

                        "probabilities": {
                            "real": round(
                                float(
                                    prediction[
                                        "real_probability"
                                    ]
                                ),
                                6,
                            ),

                            "fake": round(
                                float(
                                    prediction[
                                        "fake_probability"
                                    ]
                                ),
                                6,
                            ),
                        },
                    }
                )

        if total_effective_weight <= 0:
            raise RuntimeError(
                "Tahmin ağırlıkları hesaplanamadı."
            )

        combined_real_probability = (
            weighted_real_sum
            / total_effective_weight
        )

        combined_fake_probability = (
            weighted_fake_sum
            / total_effective_weight
        )

        combined_probability_sum = (
            combined_real_probability
            + combined_fake_probability
        )

        if combined_probability_sum <= 0:
            raise RuntimeError(
                "Birleştirilmiş tahmin olasılıkları "
                "geçersiz."
            )

        # Yuvarlama veya kayan nokta farklarına karşı
        # olasılıkları tekrar 1 toplamına getirir.
        combined_real_probability = (
            combined_real_probability
            / combined_probability_sum
        )

        combined_fake_probability = (
            combined_fake_probability
            / combined_probability_sum
        )

        if (
            combined_fake_probability
            > combined_real_probability
        ):
            predicted_id = 1

        else:
            predicted_id = 0

        predicted_label = self._get_label_name(
            predicted_id
        )

        confidence = max(
            combined_real_probability,
            combined_fake_probability,
        )

        is_chunked = len(
            chunks
        ) > 1

        return {
            "label": predicted_label,

            "label_id": predicted_id,

            "confidence": round(
                float(confidence),
                6,
            ),

            "probabilities": {
                "real": round(
                    float(
                        combined_real_probability
                    ),
                    6,
                ),

                "fake": round(
                    float(
                        combined_fake_probability
                    ),
                    6,
                ),
            },

            # Eski Chrome paneliyle uyumluluk için korunur.
            "processed_token_count": (
                total_token_count
            ),

            "maximum_token_length": (
                self.MAX_LENGTH
            ),

            "total_token_count": (
                total_token_count
            ),

            "chunk_count": len(
                chunks
            ),

            "is_chunked": is_chunked,

            "all_text_analyzed": True,

            "chunk_overlap_tokens": (
                self.CHUNK_OVERLAP_TOKENS
                if is_chunked
                else 0
            ),

            "chunks": chunk_results,
        }

    # ========================================================
    # HEALTH DURUMU
    # ========================================================

    def get_status(self) -> dict[str, Any]:
        """
        /health endpoint'inde kullanılacak model durumunu döndürür.
        """

        gpu_name = None

        if torch.cuda.is_available():
            gpu_name = (
                torch.cuda.get_device_name(0)
            )

        return {
            "loaded": self.is_loaded,

            "model_name": (
                "Turkish ELECTRA Base"
            ),

            "model_path": str(
                self.model_dir
            ),

            "device": str(
                self.device
            ),

            "gpu": gpu_name,

            "labels": {
                "0": "real",
                "1": "fake",
            },

            "minimum_token_length": (
                self.MIN_TOKEN_COUNT
            ),

            "maximum_token_length": (
                self.MAX_LENGTH
            ),

            "chunking_enabled": True,

            "chunk_overlap_tokens": (
                self.CHUNK_OVERLAP_TOKENS
            ),
        }


# ============================================================
# UYGULAMA BOYUNCA KULLANILACAK TEK MODEL SERVİSİ
# ============================================================

model_service = ModelService()