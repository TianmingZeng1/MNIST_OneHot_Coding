# 分类交叉熵代码的理解

这个函数定义了一个计算 **分类交叉熵损失（Categorical Cross Entropy Loss）** 的功能，主要用于衡量模型预测的概率分布与真实标签之间的差异。它接受三个参数：

- `params`：模型的参数，包含神经网络中的权重和偏置。
- `x`：输入数据，作为特征输入到模型中。
- `y_true`：真实标签，采用 one-hot 编码表示。

首先，通过 `model.apply(params, x)` 进行前向传播，得到每个样本的预测概率分布 `y_pred`。

接下来，计算每个样本的交叉熵损失：

```python
loss_per_example = -jnp.sum(y_true * jnp.log(y_pred + 1e-8), axis=1)
```

- **`y_true * jnp.log(y_pred + 1e-8)`**：该操作用于计算正确类别的对数概率。为了防止 `y_pred` 中的值为 `0` 导致取对数时出现数值不稳定的情况，添加了一个小数 `1e-8`。
- **`jnp.sum(..., axis=1)`**：对每个样本的所有类别的损失进行求和，得到每个样本的整体损失值。因此，`loss_per_example` 表示每个样本的损失值。

最后，通过以下代码返回所有样本的平均损失：

```python
return jnp.mean(loss_per_example)
```

- **`jnp.mean(loss_per_example)`**：计算所有样本的平均损失。我们使用平均值作为损失的返回结果，因为我们希望通过优化整个批次的平均损失来提高模型整体的准确性。

### 数学解释
分类交叉熵用于衡量两个概率分布之间的距离：一个是模型预测的概率分布 `y_pred`，另一个是真实的类别标签 `y_true`。其数学公式为：

\[
L = -\sum_{i} y_{	ext{true},i} \cdot \log(y_{	ext{pred},i})
\]

- **`y_true`** 是一个 one-hot 编码向量，其中正确类别的位置为 `1`，其他位置为 `0`。
- **`y_pred`** 是模型预测的概率分布。
- 当模型对正确类别的预测概率越接近 `1` 时，损失越小。通过最小化这个损失，模型的预测准确性不断提高。

### 总结
该函数用于计算分类交叉熵损失，以衡量模型预测的概率分布与真实标签之间的差异。它使用三个输入：模型参数（`params`）、输入数据（`x`）和真实标签（`y_true`）。首先通过模型进行前向传播得到预测概率，然后计算每个样本的交叉熵损失，最后返回所有样本的平均损失。通过最小化平均损失，模型能够不断改进其预测能力。






# Erklärung des Codes zur Berechnung der kategorischen Kreuzentropie

Diese Funktion definiert eine Berechnung der **kategorischen Kreuzentropie (Categorical Cross Entropy Loss)**, die verwendet wird, um die Differenz zwischen der von einem Modell vorhergesagten Wahrscheinlichkeitsverteilung und den tatsächlichen Labels zu messen. Die Funktion nimmt drei Parameter entgegen:

- `params`: Dies sind die Modellparameter, die die Gewichte und Biases des neuronalen Netzwerks enthalten.
- `x`: Dies sind die Eingabedaten, die als Features an das Modell übergeben werden.
- `y_true`: Dies sind die echten Labels, die in One-Hot-Codierung vorliegen.

Zuerst wird mit `model.apply(params, x)` ein Vorwärtsdurchlauf durch das Modell durchgeführt, um die Wahrscheinlichkeitsverteilung `y_pred` für jede Probe zu erhalten.

Dann wird die Kreuzentropie für jedes Beispiel berechnet:

```python
loss_per_example = -jnp.sum(y_true * jnp.log(y_pred + 1e-8), axis=1)
```

- **`y_true * jnp.log(y_pred + 1e-8)`**: Diese Operation berechnet die logarithmierte Wahrscheinlichkeit der korrekten Kategorie. Das `1e-8` wird hinzugefügt, um numerische Instabilität zu vermeiden, die auftritt, wenn `y_pred` den Wert `0` annimmt.
- **`jnp.sum(..., axis=1)`**: Diese Funktion summiert die Verluste über alle Kategorien hinweg für jede einzelne Probe, sodass `loss_per_example` den Verlustwert für jedes Beispiel enthält.

Schließlich wird der durchschnittliche Verlust aller Proben berechnet und zurückgegeben:

```python
return jnp.mean(loss_per_example)
```

- **`jnp.mean(loss_per_example)`**: Diese Berechnung ermittelt den Durchschnittswert des Verlusts aller Beispiele. Der Mittelwert wird verwendet, da wir das Modell auf Grundlage des Durchschnittsverlusts über alle Proben optimieren möchten, um die Genauigkeit insgesamt zu verbessern.

### Mathematische Erklärung
Die kategorische Kreuzentropie misst die Distanz zwischen zwei Wahrscheinlichkeitsverteilungen: einerseits der Vorhersageverteilung `y_pred` und andererseits der echten Kategorien `y_true`. Die mathematische Formel lautet:

\[
L = -\sum_{i} y_{	ext{true},i} \cdot \log(y_{	ext{pred},i})
\]

- **`y_true`** ist ein One-Hot-codierter Vektor, wobei nur das Element der korrekten Kategorie auf `1` gesetzt ist, alle anderen auf `0`.
- **`y_pred`** ist die Wahrscheinlichkeitsverteilung, die vom Modell vorhergesagt wurde.
- Der Verlust ist geringer, wenn die Vorhersagewahrscheinlichkeit für die korrekte Kategorie nahe bei `1` liegt. Durch Minimierung dieses Verlusts verbessert sich das Modell in Bezug auf seine Vorhersagefähigkeit.

### Zusammenfassung für die Erklärung 
Diese Funktion berechnet die kategorische Kreuzentropie, um die Differenz zwischen den Vorhersagen des Modells und den echten Labels zu messen. Sie nutzt dabei drei Eingaben: die Modellparameter (`params`), die Eingabedaten (`x`) und die echten Labels (`y_true`). Der Vorwärtsdurchlauf des Modells liefert die Vorhersage (`y_pred`), aus der dann die Kreuzentropie für jedes Beispiel berechnet wird. Die Kreuzentropie ermöglicht es, die Ähnlichkeit zwischen der vorhergesagten und der echten Wahrscheinlichkeitsverteilung zu messen. Schließlich wird der durchschnittliche Verlust aller Beispiele zurückgegeben, um das Modell insgesamt zu optimieren und seine Vorhersagegenauigkeit zu verbessern.


