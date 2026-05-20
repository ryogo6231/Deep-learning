# Deep Learning

## 目次

1. [ディープラーニングとは](#1-ディープラーニングとは)
2. [ディープラーニングモデルの種類](#2-ディープラーニングモデルの種類)
   - [2.1 畳み込みニューラルネットワーク（CNN）](#21-畳み込みニューラルネットワークcnn)
   - [2.2 リカレントニューラルネットワーク（RNN）](#22-リカレントニューラルネットワークrnn)
   - [2.3 トランスフォーマーモデル](#23-トランスフォーマーモデル)
3. [参考文献](#3-参考文献)

---

## 1. ディープラーニングとは

ニューラルネットワークの**隠れ層を増やす**ことで、層は深くなります。  
こうした深い層を持つネットワークのことを**ディープニューラルネットワーク**（Deep Neural Network, DNN）と呼びます。  
DNNを学習する手法のことを総じて**ディープラーニング**あるいは**深層学習**と呼びます。

### なぜ深くするのか

浅いネットワークでも理論上は任意の関数を近似できますが、層を深くすることで

- より**複雑な特徴**を段階的に学習できる
- 少ないパラメータで**高い表現力**を得られる
- 画像・音声・言語など多様なデータに対応できる

という利点があります。

### 学習の仕組み

DNNは**誤差逆伝播法**（Backpropagation）と**勾配降下法**を組み合わせて学習します。

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

- $L$：損失関数（予測と正解のずれ）
- $w$：重みパラメータ
- $\eta$：学習率（Learning Rate）

---

## 2. ディープラーニングモデルの種類

### 2.1 畳み込みニューラルネットワーク（CNN）

**Convolutional Neural Network** — 主に画像認識に使われるモデル。

**特徴**  
フィルタ（カーネル）を入力上でスライドさせる**畳み込み演算**によって、局所的な特徴（エッジ・形・テクスチャ）を段階的に抽出する。

$$
Y[i,j] = \sum_{m}\sum_{n} X[i+m,\, j+n] \cdot K[m,n]
$$

**主なレイヤー**

| レイヤー | 役割 |
|----------|------|
| 畳み込み層（Conv） | フィルタで局所特徴を抽出 |
| プーリング層（Pooling） | 特徴マップを縮小・圧縮 |
| 全結合層（FC） | 最終的なクラス分類 |

**代表的なモデル**：VGG / ResNet / EfficientNet

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.fc    = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

---

### 2.2 リカレントニューラルネットワーク（RNN）

**Recurrent Neural Network** — テキスト・音声・時系列データに使われるモデル。

**特徴**  
前の時刻の**隠れ状態** $h_{t-1}$ を次のステップへ引き継ぐことで、過去の情報を保持しながら系列を処理する。

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

**問題点と発展モデル**

長い系列では**勾配消失**が発生し、長距離の依存関係を学習できなくなる。  
これを解決するために以下のモデルが提案された。

| モデル | ゲート | 特徴 |
|--------|--------|------|
| RNN | なし | 構造がシンプル |
| LSTM | 入力・忘却・出力（3つ） | セル状態で長距離依存を記憶 |
| GRU | リセット・更新（2つ） | LSTMより軽量 |

**代表的な用途**：機械翻訳 / 文章生成 / 音声認識

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        return self.fc(out), hidden
```

---

### 2.3 トランスフォーマーモデル

**Transformer** — 現代のAIの中核をなすモデル。2017年にGoogleが発表。

**特徴**  
RNNのように順番に処理するのではなく、**Attention機構**によって系列全体を一度に処理する。  
これにより長距離の依存関係を効率よく学習でき、並列計算も可能になった。

**Self-Attention の計算式**

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- $Q$（Query）・$K$（Key）・$V$（Value）：入力から線形変換した行列
- $d_k$：Keyの次元数（スケーリングのため $\sqrt{d_k}$ で割る）

**アーキテクチャ**

```
入力
 └─ Embedding + Positional Encoding
      └─ Encoder（Multi-Head Attention → Feed Forward）× N層
           └─ Decoder（Masked Attention → Cross Attention → FF）× N層
                └─ 出力
```

**主要なモデル**

| モデル | 開発元 | 用途 |
|--------|--------|------|
| BERT | Google | 文章の理解・分類 |
| GPT | OpenAI | 文章生成 |
| ViT | Google | 画像認識 |
| Whisper | OpenAI | 音声認識 |

---

