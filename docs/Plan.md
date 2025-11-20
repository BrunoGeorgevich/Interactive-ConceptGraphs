# Documento 2: Plano de Implementação Técnica

## Título: Implementação da Arquitetura Híbrida (AIMSM) sobre o ConceptGraphs

Este documento detalha as etapas de implementação, modificações de código e estruturas de classes necessárias para refatorar o `ConceptGraphs` na arquitetura adaptativa proposta.

### 1. Visão Geral da Arquitetura

O padrão central será uma combinação de **Facade**, **Abstract Factory** e **Strategy**.

  * **`AdaptiveInferenceManager` (Facade):** A única classe com a qual o `rerun_realtime_mapping.py` irá interagir. Ela esconde toda a complexidade.
  * **`StrategySwitcher`:** Lógica interna do *Manager* que consulta o `SystemContext` para decidir qual estratégia usar.
  * **`I...Strategy` (Interfaces):** Os "contratos" que as estratégias locais e nuvem devem seguir (ex: `IObjectDetector`, `IVLM`).
  * **`Local...Strategy` / `Cloud...Strategy` (Estratégias Concretas):** As implementações reais que carregam modelos locais (YOLO, SAM) ou fazem chamadas de API (OpenAI, APIs de nuvem).


```
[rerun_realtime_mapping.py]
    |
    v
[AdaptiveInferenceManager] (Facade)
    |
    +--> [SystemContext] (Verifica rede, ambiente)
    |
    +--> [StrategySwitcher] (Decide qual estratégia usar)
    |
    +--> [IObjectDetector]
    |      |
    |      +-- [LocalDetector] (YOLO)
    |      '-- [CloudDetector] (API)
    |
    +--> [ISegmenter]
    |      |
    |      +-- [LocalSegmenter] (SAM)
    |      '-- [CloudSegmenter] (API)
    |
    +--> [IFeatureExtractor]
    |      |
    |      +-- [LocalFeatureExtractor] (OpenCLIP) <--- Sempre usado
    |      '-- [CloudFeatureExtractor] (API) <--- Retorna tensores
    |
    '--> [IVLM]
          |
          +-- [LMStudioVLM] (Ollama/GGUF)
          '-- [CloudVLM] (OpenAI/Gemini)
```

*(Nota: Conforme a nossa discussão, `IFeatureExtractor` pode ter ambas as estratégias. A `CloudFeatureExtractor` apenas computa na nuvem e retorna um tensor, que é então usado localmente).*

### 2. Fase 1: Criação do Módulo `conceptgraph/inference/`

Criar uma nova pasta `conceptgraph/inference/`.

#### Ação 2.1: Definir as Interfaces (`inference/interfaces.py`)

Este arquivo define os contratos, inspirados no `AIMSM/src/AIModules/AIModule.py`.

```python
# Em: conceptgraph/inference/interfaces.py
from abc import ABC, abstractmethod
import numpy as np
import torch
from supervision import Detections

class IInferenceStrategy(ABC):
    """Interface base, inspirada no AIModule.py do AIMSM."""
    
    @abstractmethod
    def load_model(self):
        """Carrega o modelo na memória (CPU/GPU)."""
        pass

    @abstractmethod
    def unload_model(self):
        """Libera o modelo da memória."""
        pass

    @abstractmethod
    def get_type(self) -> str:
        """Retorna 'local' ou 'cloud'."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado."""
        pass

class IObjectDetector(IInferenceStrategy):
    @abstractmethod
    def set_classes(self, classes: list[str]):
        """Define as classes que o detector deve procurar."""
        pass
    
    @abstractmethod
    def detect(self, image_path: str, image_np: np.ndarray) -> Detections:
        """Executa a detecção e retorna objetos supervision.Detections (sem máscaras)."""
        pass

class ISegmenter(IInferenceStrategy):
    @abstractmethod
    def segment(self, image_path: str, image_np: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
        """Recebe caixas e retorna máscaras torch.Tensor."""
        pass

class IFeatureExtractor(IInferenceStrategy):
    @abstractmethod
    def extract_features(self, image_np: np.ndarray, detections: Detections, classes: list[str]) -> tuple:
        """Retorna (image_crops, image_feats, text_feats) como tensores."""
        pass

class IVLM(IInferenceStrategy):
    """Interface para todos os serviços de VLM e LLM."""
    @abstractmethod
    def get_relations(self, annotated_image_path: str, labels: list[str]) -> list:
        """VLM de Mapeamento: Extrai relações espaciais."""
        pass
  
    @abstractmethod
    def get_captions(self, annotated_image_path: str, labels: list[str]) -> list:
        """VLM de Mapeamento: Extrai legendas dos objetos."""
        pass
  
    @abstractmethod
    def get_room_data(self, image_path: str, context: list) -> dict:
        """VLM de Mapeamento: Classifica o ambiente."""
        pass

    @abstractmethod
    def consolidate_captions(self, captions: list) -> str:
        """VLM de Mapeamento: Consolida legendas."""
        pass
    
    @abstractmethod
    def query_map(self, query: str, map_context: str, tools: list) -> str:
        """LLM de Inferência: Responde a perguntas do usuário sobre o mapa."""
        pass
```

#### Ação 2.2: Encapsular Estratégias Locais (`inference/local_strategies.py`)

Isto é principalmente uma refatoração do código existente em `rerun_realtime_mapping.py` e `utils/model_utils.py`.

```python
# Em: conceptgraph/inference/local_strategies.py
from .interfaces import IObjectDetector, ISegmenter, IFeatureExtractor
from ultralytics import YOLO, SAM
import open_clip
import supervision as sv
import numpy as np
import torch
from conceptgraph.utils.model_utils import compute_clip_features_batched

class LocalDetector(IObjectDetector):
    def __init__(self, checkpoint_path: str, device: str):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model: YOLO = None

    def load_model(self):
        if not self.is_loaded():
            # Esta linha é movida do rerun_realtime_mapping.py
            self.model = YOLO(self.checkpoint_path).to(self.device)

    def unload_model(self):
        if self.is_loaded():
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool: return self.model is not None
    def get_type(self) -> str: return "local"

    def set_classes(self, classes: list[str]):
        self.load_model()
        self.model.set_classes(classes)

    def detect(self, image_path: str, image_np: np.ndarray) -> Detections:
        self.load_model()
        results = self.model.predict(image_path, conf=0.5, verbose=False)
        
        # Lógica de conversão movida de rerun_realtime_mapping.py
        if not results or len(results[0].boxes) == 0:
            return sv.Detections.empty()
        
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        class_id = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # A máscara será preenchida pelo Segmenter
        empty_masks = np.empty((len(xyxy), *image_np.shape[:2]), dtype=bool) 
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id, mask=empty_masks)

# --- Classe LocalSegmenter (SAM) ---
class LocalSegmenter(ISegmenter):
    def __init__(self, checkpoint_path: str, device: str):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.predictor: SAM = None

    def load_model(self):
        if not self.is_loaded():
            self.predictor = SAM(self.checkpoint_path).to(self.device)

    def unload_model(self):
        if self.is_loaded():
            del self.predictor
            self.predictor = None
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool: return self.predictor is not None
    def get_type(self) -> str: return "local"

    def segment(self, image_path: str, image_np: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
        self.load_model()
        sam_out = self.predictor.predict(image_path, bboxes=boxes, verbose=False)
        return sam_out[0].masks.data

# --- Classe LocalFeatureExtractor (CLIP) ---
class LocalFeatureExtractor(IFeatureExtractor):
    def __init__(self, model_name: str, pretrained: str, device: str):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.clip_model = self.clip_preprocess = self.clip_tokenizer = None

    def load_model(self):
        if not self.is_loaded():
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.model_name, self.pretrained
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_tokenizer = open_clip.get_tokenizer(self.model_name)

    def unload_model(self):
        del self.clip_model, self.clip_preprocess, self.clip_tokenizer
        self.clip_model = self.clip_preprocess = self.clip_tokenizer = None
        torch.cuda.empty_cache()

    def is_loaded(self) -> bool: return self.clip_model is not None
    def get_type(self) -> str: return "local"
    
    def extract_features(self, image_np: np.ndarray, detections: Detections, classes: list[str]) -> tuple:
        self.load_model()
        # Esta função já existe em utils/model_utils.py
        return compute_clip_features_batched(
            image_np, detections, self.clip_model, self.clip_preprocess,
            self.clip_tokenizer, classes, self.device
        )

# --- Classe LMStudioVLM (Placeholder) ---
class LMStudioVLM(IVLM):
    def __init__(self, cfg):
        self.client = None # TODO: Implementar Ollama/Llama.cpp
    
    def load_model(self): pass
    def unload_model(self): pass
    def is_loaded(self) -> bool: return self.client is not None
    def get_type(self) -> str: return "local"

    def get_relations(self, *args, **kwargs) -> list:
        logging.warning("LMStudioVLM.get_relations não implementado. Retornando [].")
        return []
    def get_captions(self, *args, **kwargs) -> list:
        logging.warning("LMStudioVLM.get_captions não implementado. Retornando [].")
        return []
    def get_room_data(self, *args, **kwargs) -> dict:
        logging.warning("LMStudioVLM.get_room_data não implementado. Retornando {}.")
        return {"room_class": "N/A", "room_description": "VLM Local não implementado."}
    def consolidate_captions(self, captions: list) -> str:
        logging.warning("LMStudioVLM.consolidate_captions não implementado.")
        return "Consolidação local não implementada."
    def query_map(self, *args, **kwargs) -> str:
        logging.warning("LMStudioVLM.query_map não implementado.")
        return "VLM Local não implementado."
```

#### Ação 2.3: Implementar Estratégias de Nuvem (`inference/cloud_strategies.py`)

Esta classe irá encapsular as chamadas de API. Para a `CloudVLM`, ela reutilizará a lógica de `utils/vlm.py`.

```python
# Em: conceptgraph/inference/cloud_strategies.py
from .interfaces import IObjectDetector, ISegmenter, IFeatureExtractor, IVLM
from conceptgraph.utils.vlm import (
    get_vlm_openai_like_client, 
    get_obj_rel_from_image, 
    get_obj_captions_from_image, 
    get_room_data_from_image, 
    consolidate_captions
)
import httpx
import supervision as sv
import numpy as np
import torch
import base64
import io
from PIL import Image

# --- Classe CloudVLM ---
class CloudVLM(IVLM):
    def __init__(self, model: str, api_key: str, base_url: str):
        # Esta classe encapsula o código que já existe em utils/vlm.py
        self.client = get_vlm_openai_like_client(model, api_key, base_url)
    
    def load_model(self): pass # API não precisa carregar
    def unload_model(self): pass # API não precisa descarregar
    def is_loaded(self) -> bool: return True # API está sempre "carregada"
    def get_type(self) -> str: return "cloud"

    def get_relations(self, annotated_image_path: str, labels: list[str]) -> list:
        # 'httpx.ConnectError' será capturado pelo Facade (Manager)
        return get_obj_rel_from_image(self.client, annotated_image_path, labels)

    def get_captions(self, annotated_image_path: str, labels: list[str]) -> list:
        return get_obj_captions_from_image(self.client, annotated_image_path, labels)
  
    def get_room_data(self, image_path: str, context: list) -> dict:
        return get_room_data_from_image(self.client, image_path, context)
    
    def consolidate_captions(self, captions: list) -> str:
        return consolidate_captions(self.client, captions)
    
    def query_map(self, query: str, map_context: str, tools: list) -> str:
        # TODO: Implementar lógica de consulta ao LLM da nuvem (Estágio 2)
        pass

# --- Classe CloudDetector (Exemplo de Implementação) ---
class CloudDetector(IObjectDetector):
    def __init__(self, api_url: str, api_key: str, device: str):
        self.api_url = api_url
        self.client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=20.0)
        self.device = device
        self.classes = []

    def load_model(self): pass
    def unload_model(self): pass
    def is_loaded(self) -> bool: return True
    def get_type(self) -> str: return "cloud"
    
    def set_classes(self, classes: list[str]):
        self.classes = classes
        # (Opcional: enviar classes para a API, se ela suportar)
        # self.client.post(f"{self.api_url}/set_classes", json={"classes": classes}, timeout=5)

    def detect(self, image_path: str, image_np: np.ndarray) -> Detections:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = self.client.post(f"{self.api_url}/detect", files=files, data={"classes": self.classes})
        
        response.raise_for_status() # Lança exceção se for 4xx ou 5xx
        data = response.json() # Espera-se: {'xyxy': [[..]], 'confidence': [..], 'class_id': [..]}
        
        xyxy = np.array(data['xyxy'])
        conf = np.array(data['confidence'])
        class_id = np.array(data['class_id'], dtype=int)
        empty_masks = np.empty((len(xyxy), *image_np.shape[:2]), dtype=bool) 
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id, mask=empty_masks)

# --- Classe CloudFeatureExtractor (Exemplo de Implementação) ---
class CloudFeatureExtractor(IFeatureExtractor):
    def __init__(self, api_url: str, api_key: str, device: str):
        self.api_url = api_url
        self.client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=20.0)
        self.device = device

    def load_model(self): pass
    def unload_model(self): pass
    def is_loaded(self) -> bool: return True
    def get_type(self) -> str: return "cloud"

    def extract_features(self, image_np: np.ndarray, detections: Detections, classes: list[str]) -> tuple:
        image_crops = []
        files_to_send = []
        
        pil_image = Image.fromarray(image_np)
        padding = 20
        
        # Preparar crops (lógica de model_utils.py)
        for idx in range(len(detections.xyxy)):
            x_min, y_min, x_max, y_max = detections.xyxy[idx]
            # ... (lógica de padding) ...
            cropped_image_pil = pil_image.crop((x_min, y_min, x_max, y_max))
            image_crops.append(cropped_image_pil)
            
            # Converter para bytes e adicionar à lista de arquivos
            img_byte_arr = io.BytesIO()
            cropped_image_pil.save(img_byte_arr, format='PNG')
            files_to_send.append(('images', img_byte_arr.getvalue()))

        if not files_to_send:
            return [], np.array([]), np.array([])
        
        # Enviar lote de imagens para a API
        response = self.client.post(f"{self.api_url}/extract_features", files=files_to_send)
        response.raise_for_status()
        data = response.json() # Espera-se: {'image_feats': [[...], [...]]}
        
        # Converter para tensor (conforme discutido, o tensor deve estar localmente)
        image_feats_tensor = torch.tensor(data['image_feats'], dtype=torch.float, device=self.device)
        
        # (Nota: A API do CLIP pode não precisar das 'classes', então text_feats fica vazio ou é tratado de outra forma)
        text_feats_np = np.array([]) 
        
        return image_crops, image_feats_tensor.cpu().numpy(), text_feats_np
```

### 3. Fase 2: Implementação do Núcleo de Chaveamento (AIMSM)

#### Ação 3.1 & 3.2: Contexto e Seletor (`inference/switcher.py`)

Isto é inspirado no `CSSM.py`.

```python
# Em: conceptgraph/inference/switcher.py
import httpx
import logging
from .interfaces import IInferenceStrategy

class SystemContext:
    """Armazena o estado atual do robô. (Inspirado no CSSM)"""
    
    def __init__(self):
        self.connectivity_status = "offline"
        self.environment_profile = "indoor" # Default
        self.update() # Verifica o estado inicial
    
    def update(self):
        self.connectivity_status = self._check_network()
        # (No futuro, 'update' também pode verificar o perfil do ambiente)
    
    def _check_network(self, timeout=2.0) -> str:
        """Verifica a conectividade com a Internet."""
        try:
            httpx.get("http://www.google.com", timeout=timeout)
            logging.debug("Checagem de rede: Online")
            return "online"
        except (httpx.ConnectError, httpx.Timeout, httpx.RequestError):
            logging.debug("Checagem de rede: Offline")
            return "offline"

class StrategySwitcher:
    """Seleciona a estratégia apropriada com base no contexto."""
    
    def __init__(self, preferred_mode: str = "online"):
        self.preferred_mode = preferred_mode

    def select_strategy(self, context: SystemContext, 
                          cloud_strategy: IInferenceStrategy, 
                          local_strategy: IInferenceStrategy) -> IInferenceStrategy:
        
        # 1. Seleção baseada na preferência e conectividade
        strategy_by_network = local_strategy # Default é local
        if self.preferred_mode == "online":
            if context.connectivity_status == "online":
                if cloud_strategy is not None:
                    strategy_by_network = cloud_strategy
                else:
                    logging.warning("Modo online preferido, mas estratégia de nuvem não está configurada. Usando local.")
            else:
                logging.debug("Modo online preferido, mas rede offline. Usando fallback local.")
    
        # 2. Seleção baseada no perfil do ambiente (Indoor/Outdoor)
        # (Esta lógica será mais complexa quando você tiver mais de um modelo local/nuvem por perfil)
        # Por agora, apenas retornamos a estratégia selecionada pela rede.
        
        final_strategy = strategy_by_network
        
        # 3. TODO: Implementar o chaveamento de PROMPT/CLASSES aqui
        if hasattr(final_strategy, 'set_classes'):
            if context.environment_profile == "indoor":
                final_strategy.set_classes(self.indoor_classes)
            elif context.environment_profile == "outdoor":
                final_strategy.set_classes(self.outdoor_classes)
        
        return final_strategy
```

*(Nota: O `StrategySwitcher` precisará ter acesso às listas de classes (indoor/outdoor) e prompts, que devem ser passados pelo `Manager`.)*

#### Ação 3.3: O Gerenciador/Facade (`inference/manager.py`)

Esta é a classe principal, inspirada no `AIMSM.py`.

```python
# Em: conceptgraph/inference/manager.py
import logging
import torch
from omegaconf import DictConfig
from .interfaces import *
from .local_strategies import LocalDetector, LocalSegmenter, LocalFeatureExtractor, LMStudioVLM
from .cloud_strategies import CloudDetector, CloudSegmenter, CloudFeatureExtractor, CloudVLM
from .switcher import SystemContext, StrategySwitcher

class AdaptiveInferenceManager:
    def __init__(self, cfg_inference: DictConfig, cfg_classes: DictConfig, device: str):
        self.cfg_inf = cfg_inference
        self.cfg_cls = cfg_classes
        self.device = device
        
        self.context = SystemContext()
        self.switcher = StrategySwitcher(preferred_mode=self.cfg_inf.mode)
        
        # Carregar listas de classes para chaveamento de ambiente
        self._load_profile_configs()
        
        # Inicializar todas as estratégias (locais e nuvem)
        self._init_strategies()
        
        # Ponteiros para as estratégias ATIVAS
        self.active_detector: IObjectDetector = None
        self.active_segmenter: ISegmenter = None
        self.active_features: IFeatureExtractor = None
        self.active_vlm: IVLM = None
        
        # Forçar uma atualização inicial para carregar os modelos corretos
        self.update_active_strategies() 

    def _init_strategies(self):
        # Instancia estratégias locais (sempre necessárias para fallback)
        cfg_local = self.cfg_inf.offline_models
        self.local_detector = LocalDetector(cfg_local.yolo_path, self.device)
        self.local_segmenter = LocalSegmenter(cfg_local.sam_path, self.device)
        self.local_features = LocalFeatureExtractor(
            cfg_local.clip_model_name, cfg_local.clip_pretrained, self.device
        )
        self.local_vlm = LMStudioVLM(cfg_local.get("vlm_config", None))

        # Instancia estratégias da nuvem (se configuradas)
        self.cloud_detector = self.cloud_segmenter = self.cloud_features = self.cloud_vlm = None
        if "online_models" in self.cfg_inf and self.cfg_inf.online_models is not None:
            cfg_cloud = self.cfg_inf.online_models
            
            # (Descomentar quando as APIs estiverem prontas)
            # self.cloud_detector = CloudDetector(cfg_cloud.yolo_api, cfg_cloud.api_key, self.device)
            # self.cloud_segmenter = CloudSegmenter(cfg_cloud.sam_api, cfg_cloud.api_key, self.device)
            # self.cloud_features = CloudFeatureExtractor(cfg_cloud.clip_api, cfg_cloud.api_key, self.device)
            
            self.cloud_vlm = CloudVLM(
                cfg_cloud.vlm_model, cfg_cloud.vlm_api_key, cfg_cloud.vlm_base_url
            )
        
        # Fallback para estratégias não implementadas na nuvem
        if self.cloud_detector is None: self.cloud_detector = self.local_detector
        if self.cloud_segmenter is None: self.cloud_segmenter = self.local_segmenter
        if self.cloud_features is None: self.cloud_features = self.local_features
        if self.cloud_vlm is None: self.cloud_vlm = self.local_vlm

    def _load_profile_configs(self):
        # Carregar classes e prompts para diferentes perfis (indoor/outdoor)
        # Esta é a lógica de adaptabilidade de ambiente
        
        # Carrega classes indoor (o padrão do ConceptGraphs)
        with open(self.cfg_cls.classes_file, "r") as f:
            all_classes = [cls.strip() for cls in f.readlines()]
        if self.cfg_cls.skip_bg:
            self.indoor_classes = [cls for cls in all_classes if cls not in self.cfg_cls.bg_classes]
        else:
            self.indoor_classes = all_classes
        self.indoor_vlm_prompt = SYSTEM_PROMPT_ONLY_TOP # (Exemplo)

        # TODO: Carregar classes e prompts "outdoor" de novos arquivos de config
        self.outdoor_classes = ["car", "person", "tree", "building", "road"] # (Exemplo)
        self.outdoor_vlm_prompt = "..." # (Um prompt focado em navegação)

    def update_active_strategies(self):
        """
        Atualiza o contexto (rede) e, em seguida, seleciona e gerencia (load/unload)
        as estratégias ativas para cada serviço.
        """
        self.context.update() # Verifica a conectividade
        
        service_map = {
            "detector": (self.cloud_detector, self.local_detector, self.active_detector),
            "segmenter": (self.cloud_segmenter, self.local_segmenter, self.active_segmenter),
            "features": (self.cloud_features, self.local_features, self.active_features),
            "vlm": (self.cloud_vlm, self.local_vlm, self.active_vlm), 
        }
        
        active_pointers = {}
        
        for service_type, (cloud_strat, local_strat, active_strat) in service_map.items():
            if local_strat is None: continue # Serviço não implementado
            
            # Passar as classes/prompts relevantes para o switcher
            if self.context.environment_profile == "indoor":
                self.switcher.active_classes = self.indoor_classes
                self.switcher.active_prompt = self.indoor_vlm_prompt
            else:
                self.switcher.active_classes = self.outdoor_classes
                self.switcher.active_prompt = self.outdoor_vlm_prompt
            
            desired_strategy = self.switcher.select_strategy(self.context, cloud_strat, local_strat)
            
            if active_strat is not desired_strategy:
                if active_strat is not None and active_strat.is_loaded():
                    logging.info(f"Descarregando {service_type} ({active_strat.get_type()})")
                    active_strat.unload_model()
                
                active_strat = desired_strategy
                if not active_strat.is_loaded():
                    logging.info(f"Carregando {service_type} ({active_strat.get_type()})")
                    active_strat.load_model()
            
            active_pointers[service_type] = active_strat
        
        # Atualiza os ponteiros de classe
        self.active_detector = active_pointers["detector"]
        self.active_segmenter = active_pointers["segmenter"]
        self.active_features = active_pointers["features"]
        self.active_vlm = active_pointers["vlm"]

    # --- Métodos Públicos (A Interface da Facade) ---

    def detect(self, image_path: str, image_np: np.ndarray) -> Detections:
        try:
            return self.active_detector.detect(image_path, image_np)
        except (httpx.ConnectError, httpx.Timeout, httpx.HTTPStatusError) as e:
            logging.warning(f"Detecção na nuvem falhou ({e}), fazendo fallback para local.")
            self.active_detector.unload_model()
            self.active_detector = self.local_detector
            self.active_detector.load_model()
            return self.active_detector.detect(image_path, image_np)

    def segment(self, image_path: str, image_np: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
        try:
            return self.active_segmenter.segment(image_path, image_np, boxes)
        except (httpx.ConnectError, httpx.Timeout, httpx.HTTPStatusError) as e:
            logging.warning(f"Segmentação na nuvem falhou ({e}), fazendo fallback para local.")
            self.active_segmenter.unload_model()
            self.active_segmenter = self.local_segmenter
            self.active_segmenter.load_model()
            return self.active_segmenter.segment(image_path, image_np, boxes)
    
    def extract_features(self, image_np: np.ndarray, detections: Detections, classes: list[str]) -> tuple:
        try:
            return self.active_features.extract_features(image_np, detections, classes)
        except (httpx.ConnectError, httpx.Timeout, httpx.HTTPStatusError) as e:
            logging.warning(f"Extração de features na nuvem falhou ({e}), fazendo fallback para local.")
            self.active_features.unload_model()
            self.active_features = self.local_features
            self.active_features.load_model()
            return self.active_features.extract_features(image_np, detections, classes)
    
    def get_vlm(self) -> IVLM:
        """Retorna a estratégia VLM ativa. O fallback é tratado dentro de cada chamada VLM."""
        return self.active_vlm

    def set_environment_profile(self, profile: str):
        """Gatilho externo para mudar o contexto (ex: de 'indoor' para 'outdoor')."""
        if profile != self.context.environment_profile:
            logging.info(f"Mudando perfil do ambiente para: {profile}")
            self.context.environment_profile = profile
            # Força uma re-seleção de estratégia no próximo frame
            self.update_active_strategies()
```

### 4. Fase 3: Refatoração do `rerun_realtime_mapping.py`

Esta é a etapa final, onde o impacto no código principal é minimizado.

```python
# Em: conceptgraph/slam/rerun_realtime_mapping.py

# --- REMOVER ESTAS IMPORTAÇÕES ---
# from ultralytics import YOLO, SAM, FastSAM
# import open_clip
# from conceptgraph.utils.vlm import get_vlm_openai_like_client

# --- ADICIONAR ESTAS IMPORTAÇÕES ---
from conceptgraph.inference.manager import AdaptiveInferenceManager
from conceptgraph.utils.general_utils import ObjectClasses # Manter
from conceptgraph.utils.vlm import make_vlm_edges_and_captions # Manter por enquanto

@hydra.main(...)
def main(cfg: DictConfig):
    # ... (inicialização do tracker, orr, owandb) ...
    cfg = process_cfg(cfg)

    # =========================
    # Model Initialization (MODIFICADO)
    # =========================
    obj_classes = ObjectClasses(
        classes_file_path=cfg.classes_file,
        bg_classes=cfg.bg_classes,
        skip_bg=cfg.skip_bg,
    )
    
    # A Facade cuida de TODO o carregamento de modelos e classes
    manager = AdaptiveInferenceManager(cfg.inference, cfg.classes, cfg.device)

    # ... (inicialização do dataset, listas de objetos, renderer, paths) ...
    # A inicialização do 'openai_client' foi REMOVIDA.

    # =========================
    # Main Processing Loop Over Frames
    # =========================
    for frame_idx in trange(len(dataset)):
        # ... (lógica de 'exit_early') ...
        # ... (carregamento de 'color_path', 'image_original_pil', tensores, etc.) ...

        # IMPORTANTE: Atualiza o contexto (rede) e faz o chaveamento (load/unload)
        # Esta chamada é muito rápida se nenhum chaveamento for necessário.
        manager.update_active_strategies()

        # ... (lógica de 'orr_log_camera') ...

        # =========================
        # Detection and Segmentation (MODIFICADO)
        # =========================
        if run_detections:
            # ... (carregar imagem) ...
            
            # 1. Deteção (usando a Facade)
            # O 'manager' já sabe se usa local ou nuvem e já lidou com o fallback.
            curr_det = manager.detect(str(color_path), image_rgb)
            detection_class_labels = [ # Recria os labels que 'make_vlm_edges_and_captions' espera
                f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
                for class_idx, class_id in enumerate(curr_det.class_id)
            ]
            
            # 2. Segmentação (usando a Facade)
            if curr_det.xyxy.shape[0] > 0:
                boxes_tensor = torch.from_numpy(curr_det.xyxy).to(cfg.device)
                masks_tensor = manager.segment(str(color_path), image_rgb, boxes_tensor)
                curr_det.mask = masks_tensor.cpu().numpy()
            else:
                curr_det.mask = np.empty((0, *image_rgb.shape[:2]), dtype=bool)

            # 3. Extração de Features (usando a Facade)
            image_crops, image_feats, text_feats = manager.extract_features(
                image_rgb, curr_det, obj_classes.get_classes_arr()
            )
            
            # 4. VLM (usando a Facade)
            # Pegamos o serviço VLM ativo (que pode ser local ou nuvem)
            active_vlm = manager.get_vlm()
            
            # Ação: Modificar 'make_vlm_edges_and_captions' em utils/general_utils.py
            # para aceitar a interface 'IVLM' em vez de 'OpenAIClient'
            labels, edges, edge_image, captions, room_data = make_vlm_edges_and_captions(
                image, curr_det, obj_classes, detection_class_labels,
                det_exp_vis_path, color_path, cfg.make_edges,
                room_data_list,
                active_vlm # <--- Passa a interface VLM ativa
L           ) 
            
            # ... (A lógica de salvar 'results' permanece a mesma) ...
        
        # ... (A lógica de 'load_saved_detections' permanece a mesma) ...
        
        # ... (Toda a lógica de SLAM (filter_gobs, detections_to_obj_pcd_and_bbox, 
        #      compute_spatial_similarities, merge_obj_matches, etc.) 
        #      permanece INTACTA, pois o formato de dados 'gobs' e 'detection_list'
        #      foi preservado.) ...

    # ... (Loop termina) ...
    
    # --- Consolidação Final (MODIFICADO) ---
    active_vlm = manager.get_vlm() # Pega o VLM ativo novamente
    for obj in objects:
        obj_captions = obj["captions"][:20]
        # Tratar o caso em que o VLM local (ou VLM nenhum) está ativo
        if active_vlm is not None:
            consolidated_caption = active_vlm.consolidate_captions(obj_captions)
        else:
            consolidated_caption = "Consolidação indisponível no modo offline."
        obj["consolidated_caption"] = consolidated_caption

    # ... (Resto da lógica de 'save_pointcloud', 'save_obj_json', etc. permanece intacta) ...
```

### 5. Fase 4: Configuração (Hydra)

As configurações do Hydra serão cruciais para gerenciar os *backends*.

**Ação 5.1: Modificar `rerun_realtime_mapping.yaml`**

Adicionar a seleção do *backend* de inferência.

```yaml
# Em: conceptgraph/hydra_configs/rerun_realtime_mapping.yaml
defaults:
  - base
  - base_mapping
  - robot_at_virtual_home
  - classes
  - logging_level
  - _self_
  - inference: offline # <--- NOVO: Escolha 'offline' ou 'online'

# ... (configurações de stride, exp_suffix, etc.)

# REMOVER as configurações de SAM, pois elas vão para o config de inferência
# defaults:
#   - sam
```

**Ação 5.2: Criar Configs de Inferência**

Criar a pasta `conceptgraph/hydra_configs/inference/`.

**`inference/offline.yaml`** (Modo Padrão / Fallback)

```yaml
# Em: conceptgraph/hydra_configs/inference/offline.yaml
mode: "offline"

offline_models:
  yolo_path: "yolov8x-worldv2.pt" # (Caminho para o modelo local)
  sam_path: "sam2.1_l.pt" # (Caminho para o modelo local)
  clip_model_name: "ViT-H-14"
  clip_pretrained: "laion2b_s32b_b79k"
  vlm_config: null # (Configuração para o futuro LMStudioVLM)

online_models: null
```

**`inference/online.yaml`** (Modo Nuvem)

```yaml
# Em: conceptgraph/hydra_configs/inference/online.yaml
mode: "online"

# Modelos offline são OBRIGATÓRIOS para o fallback
offline_models:
  yolo_path: "yolov8x-worldv2.pt"
  sam_path: "sam2.1_l.pt"
  clip_model_name: "ViT-H-14"
  clip_pretrained: "laion2b_s32b_b79k"
  vlm_config: null

online_models:
  # Endpoints da sua API (Back-end)
  yolo_api: "https://seu-servidor.com/api/yolo/detect"
  sam_api: "https://seu-servidor.com/api/sam/segment"
  clip_api: "https://seu-servidor.com/api/clip/extract"
  api_key: ${oc.env:MINHA_API_KEY} # Lê da variável de ambiente

  # Configuração do VLM (já existente)
  vlm_model: "google/gemini-2.5-flash-lite"
  vlm_api_key: ${oc.env:OPENROUTER_API_KEY}
  vlm_base_url: ${oc.env:OPENROUTER_API_BASE_URL}
```

Este plano técnico implementa rigorosamente a sua visão e a arquitetura `AIMSM`, isolando as novas funcionalidades e minimizando o risco de quebrar o núcleo testado do `ConceptGraphs`.