# 🐱🐶 Classificador de Gatos e Cachorros com TensorFlow

Este projeto utiliza uma rede neural convolucional (CNN) simples para classificar imagens entre **gatos** e **cachorros**, usando o dataset "Cats vs Dogs" disponível no `tensorflow_datasets`.

---

## 📂 Estrutura do Projeto

O arquivo de código `main.py` contém todos os passos para realizar a construção, treino e uso da rede. O arquivo contém:

- `criar_modelo()` — Função que define a arquitetura da rede neural.
- `carregar_modelo()` — Função que carrega a arquitetura da rede neural salva em arquivo.
- `treinar()` — Treina o modelo com o conjunto de dados e salva o modelo treinado.
- `plot_history()` — Carrega um modelo salvo ou cria um novo se não existir.
- `salvar_history()` — Salvar dados do histórico.
- `carregar_history()` — Carregar dados do histórico.
- `visualizar_history()` — Plot dados do histórico.
- `test_image()` — Script para testar o modelo com uma imagem aleatória da base de validação.

---

## ✅ Funcionalidades

- Classificação binária de imagens (Gato ou Cachorro)
- Utilização de CNN com `Conv2D`, `MaxPooling2D` e `Dense`
- Salvamento automático do modelo treinado
- Reuso do modelo salvo sem necessidade de re-treinamento
- Visualização de predições com imagens aleatórias

---

## 🧠 Pré-requisitos

> ⚙️ Adicione aqui instruções sobre instalação de pacotes, ativação de ambiente virtual e dependências como TensorFlow e matplotlib.

---

## ▶️ Como Executar

```bash
# 1. (Instale dependências e configure o ambiente)
# 2. Treine o modelo (ou carregue um já salvo)
python seu_script_de_treino.py

# 3. Faça uma predição com imagem aleatória
python predict_random_image.py

```