# Documento 1: Plano de Doutoramento

## Título da Proposta: Uma Arquitetura Híbrida, Adaptável e Modular para Mapeamento Semântico e Interação Robótica sobre o ConceptGraphs

### 1. Introdução e Motivação

O projeto `ConceptGraphs` estabelece uma base robusta para o mapeamento semântico 3D, fundindo a percepção geométrica com um grafo de conhecimento de objetos e suas relações. No entanto, para a aplicação prática em robótica autônoma (como assistência a deficientes visuais ou robôs de serviço), o *pipeline* atual apresenta debilidades críticas:

1.  **Dependência de Conexão:** A dependência de modelos de linguagem e visão (VLMs) baseados em nuvem para extração de semântica (relações, legendas) torna o sistema vulnerável a falhas de rede, interrompendo o mapeamento.
2.  **Monolitismo de Contexto:** O sistema utiliza um conjunto fixo de modelos e classes de objetos, operando de forma idêntica independentemente do ambiente (ex: "dentro de casa" vs. "fora de casa"), o que leva à extração de semântica irrelevante ou incorreta.
3.  **Interação Limitada:** A interação foca na consulta direta ao grafo, sem capacidade de interpretar a *intenção* do usuário ou gerir diálogos complexos.

Este plano de doutoramento propõe uma nova arquitetura de *software* sobre o núcleo do `ConceptGraphs`. O objetivo é transformar o `ConceptGraphs` num sistema:

  * **Distribuído (Híbrido):** Capaz de operar com modelos de IA locais (offline) ou modelos em nuvem (online), e realizar um "chaveamento" (fallback) automático em caso de perda de conectividade.
  * **Adaptável (Context-Aware):** Capaz de alterar dinamicamente seu *pipeline* de percepção e seus *prompts* de IA com base no contexto operacional (ex: "Indoor" vs. "Outdoor"), otimizando a relevância da semântica extraída.
  * **Modular (Extensível):** Desenhado para que novos modelos, contextos e ferramentas possam ser adicionados com mínimo impacto, inspirado na arquitetura `AIMSM` (Adaptive AI Model Switching Mechanism).

A arquitetura proposta é dividida em dois estágios principais: (1) Mapeamento Semântico Híbrido e (2) Inferência e Interação Baseada em LLM.

### 2. Estágio 1: Mapeamento Semântico Híbrido e Adaptável

Este estágio estende o *pipeline* de mapeamento do `ConceptGraphs` (`rerun_realtime_mapping.py`) para torná-lo robusto e ciente do contexto.

#### 2.1. O Gerenciador de Inferência Híbrida (Núcleo AIMSM)

No coração da arquitetura estará um *Facade* (Gerenciador) que abstrai a lógica de IA do *pipeline* de SLAM. Este gerenciador será responsável por:

  * **Gerenciamento de Estratégia (Strategy Pattern):** Encapsular todos os modelos de IA (Detecção, Segmentação, VLM) em "estratégias" intercambiáveis (ex: `LocalDetector`, `CloudDetector`).
  * **Gerenciamento de Recursos (AIMSM):** Implementar a lógica de `load_model()` e `unload_model()` para carregar dinamicamente apenas os modelos necessários na VRAM, liberando recursos quando uma estratégia se torna inativa.

#### 2.2. O Chaveamento de Conectividade (Modo Distribuído)

O gerenciador irá monitorar continuamente a conectividade da rede.

  * **Modo Online (Ideal):** Por padrão, o sistema utilizará estratégias baseadas em nuvem (APIs e VLMs potentes) para todas as tarefas de IA (Detecção, Segmentação, Relações, Legendas). Isso garante a maior qualidade semântica com o menor custo computacional local.
  * **Modo Offline (Fallback):** Se a conexão de rede for perdida, o gerenciador (inspirado no `CSSM` do `AIMSM`) detecta a falha e, no *frame* seguinte, automaticamente:
    1.  Chama `unload_model()` nas estratégias de nuvem (que não fazem nada).
    2.  Chama `load_model()` nas estratégias locais (ex: `LocalDetector` (YOLO), `LocalSegmenter` (SAM) e `LMStudioVLM` (ex: LLaVA GGUF)).
  * **O Papel do CLIP:** Conforme nossa análise, os *features* do CLIP são essenciais para a lógica de *matching* e *merging* local do `ConceptGraphs`. A arquitetura suportará isso:
    1.  **Modo Offline:** O `LocalFeatureExtractor` (executando `open_clip`) será carregado e gerará os *features* localmente.
    2.  **Modo Online:** O `CloudFeatureExtractor` chamará uma API de CLIP. A API retornará o *vetor* (ex: JSON), que será então convertido num `torch.Tensor` local.
    <!-- end list -->
      * Em ambos os casos, o *pipeline* de SLAM recebe os `torch.Tensor`s que espera, permitindo que a lógica de *matching* e *merging* (que é sempre local) funcione sem alterações.

#### 2.3. O Chaveamento de Contexto (Modo Adaptável)

Esta é a segunda dimensão do chaveamento. O gerenciador manterá um "estado de perfil ambiental" (ex: `INDOOR`, `OUTDOOR`, `WORK`).

  * **Gatilho:** A mudança de estado pode ser manual (via GUI/comando) ou automática (se o `CloudVLM` reportar "indo para fora de casa" ou se o `LocalDetector` falhar em encontrar objetos *indoor* por N *frames*).
  * **Ação:** Quando o estado muda (ex: `INDOOR` -\> `OUTDOOR`):
    1.  **Detecção:** A estratégia de detecção (seja local ou nuvem) tem seu vocabulário alterado. `set_classes(["sofá", "mesa"])` é substituído por `set_classes(["carro", "árvore", "pessoa"])`.
    2.  **Semântica (VLM/LLM):** O *prompt* do sistema para o `IMappingVLM` (local ou nuvem) é substituído. O *prompt* "indoor" (focado em "on top of", "next to" para móveis) é trocado por um *prompt* "outdoor" (focado em "na frente de", "perto de" para navegação).

#### 2.4. Saída: O Mapa Semântico Vetorial

Ao final do Estágio 1, o mapa semântico 3D consolidado (nuvens de pontos, BBoxes, legendas consolidadas, e o grafo de relações) não será apenas um arquivo `.pkl.gz`. Ele será processado e indexado num banco de dados vetorial (Qdrant), preparando-o para o Estágio 2.

### 3. Estágio 2: Inferência e Interação Híbrida

Este estágio define como o usuário interage com o mapa semântico gerado. Ele é executado *após* o mapeamento (ou em paralelo, consultando o mapa em construção).

#### 3.1. Consulta Híbrida por LLM

O usuário interage com o sistema via linguagem natural (voz ou texto).

  * **LLM Híbrida:** O `AdaptiveInferenceManager` também gerenciará duas estratégias de LLM de "consulta":
    1.  `CloudLLMQuery (IVLM)`: Usa um modelo potente (ex: GPT-4o, Gemini) para consultas complexas, interpretação de intenção e diálogo.
    2.  `LocalLLMQuery (IVLM)`: Usa um modelo local (ex: Llama-3-8B GGUF) para consultas rápidas, offline, ou para tarefas simples (ex: "Onde está o micro-ondas?").
  * O chaveamento segue a mesma lógica de conectividade do Estágio 1.

#### 3.2. Interpretação de Intenção e Recuperação Semântica

O sistema irá além de consultas diretas.

  * **Intenção:** A LLM (local ou nuvem) receberá a consulta (ex: "Estou com fome").
  * **Raciocínio (Chain-of-Thought):** A LLM irá raciocinar que "fome" implica "comida", "cozinhar" ou "aquecer". Isso se traduz em objetos-alvo: "geladeira", "fogão", "micro-ondas".
  * **Consulta Vetorial:** A LLM gera *embeddings* para esses termos e consulta o Qdrant para encontrar os objetos mais relevantes no mapa.
  * **Resposta:** A LLM formula uma resposta útil (ex: "Eu encontrei uma geladeira e um micro-ondas na cozinha. O micro-ondas está a 3 metros à sua esquerda.").

#### 3.3. Diálogo e Módulos de Capacidade Própria (MCPs)

  * **Máquina de Estados de Diálogo:** Para lidar com ambiguidades (ex: "Eu encontrei duas cadeiras. Você quer a cadeira do escritório ou a cadeira da sala de jantar?"), o sistema entrará num estado de *follow-up question*.
  * **MCPs (Tools/Function Calling):** A arquitetura da LLM será estendida com "Módulos de Capacidade Própria" (MCPs), que são ferramentas de API que a LLM pode chamar.
      * *Exemplo:*
        1.  Usuário: "Leve-me ao restaurante X."
        2.  LLM: (Detecta intenção de navegação).
        3.  LLM: (Chama a `NavigationTool` com o argumento "restaurante X").
        4.  `NavigationTool` (MCP): (Faz uma chamada de API ao Google Maps, obtém latitude/longitude).
        5.  LLM: (Recebe as coordenadas e passa a informação ao sistema de navegação do robô).

### 4. Metodologia de Avaliação e Validação

A tese será validada através de uma série de experimentos comparativos entre o `ConceptGraphs` (baseline) e a nova Arquitetura Híbrida (proposta) em ambientes simulados (ex: `Robot@VirtualHome`).

1.  **Métricas de Mapeamento - Robustez:**

      * **Teste:** Executar ambos os sistemas num cenário de mapeamento e simular uma perda total de conectividade a 50% do percurso.
      * **Hipótese:** O *baseline* (usando VLM na nuvem) falhará em gerar novas relações. A arquitetura proposta fará o *fallback* para os modelos locais e completará o mapa (possivelmente com menor detalhe semântico, mas sem falha).

2.  **Métricas de Mapeamento - Adaptabilidade:**

      * **Teste:** Criar um cenário de simulação que transita de um ambiente *indoor* (`Robot@VirtualHome`) para um ambiente *outdoor* (ex: CARLA ou similar).
      * **Hipótese:** O *baseline* continuará a detectar "paredes" e "sofás" no ambiente externo, poluindo o mapa. A arquitetura proposta detectará a mudança de estado e "chaveará" os modelos, passando a detectar "carros" e "árvores" e aplicando os *prompts* corretos.

3.  **Métricas de Mapeamento - Qualidade (Baseline):**

      * **Teste:** Executar ambos os sistemas em condições ideais (conectividade total, ambiente *indoor*).
      * **Hipótese:** A qualidade do mapa semântico final (número de objetos, precisão das relações, qualidade das legendas) da arquitetura proposta (em modo online) será similar ou superior à do *baseline*.

4.  **Métricas de Inferência - Complexidade da Consulta:**

      * **Teste:** Submeter ambos os sistemas a um *dataset* de perguntas:
          * *Nível 1 (Diretas):* "Onde está a cama?"
          * *Nível 2 (Intenção):* "Estou com sono."
          * *Nível 3 (Follow-up):* "Qual delas?"
          * *Nível 4 (MCP/Tool):* "Qual é a previsão do tempo lá fora?"
      * **Hipótese:** O *baseline* só responderá ao Nível 1. A arquitetura proposta será capaz de gerir todos os níveis através da combinação de LLM (local/nuvem), Qdrant e MCPs.

### 5. Contribuição

A contribuição central desta tese é uma nova arquitetura de *software* (inspirada no `AIMSM`) que dota sistemas de mapeamento semântico como o `ConceptGraphs` de **robustez** (fallback online/offline) e **consciência contextual** (chaveamento de ambiente), tornando-os viáveis para robôs autônomos no mundo real.