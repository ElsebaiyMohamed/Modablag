# Cross-modal Contrastive Learning for Speech Translation

This is an enhanced light implementation of NAACL 2022 paper "Cross-modal Contrastive Learning for Speech Translation" [here](https://arxiv.org/abs/2205.02444)

![alt](../assests/motivation_figure.png)

---

## Model architectural components

1) Wave2vec2.0 + 2 Conv1D layer for speech modal. ![alt](../assests/wave2vec.png)

2) Word Empeeding layer for text Modal. ![alt](../assests/word%20emp.png)

3) Average Pooling layer applied over the out of 1 & 2 encoders to normlize befor calculate contrastive loss between them. ![alt](../assests/ctr%20loss.png)

4) Transformer Encoder and Decoder As Mentioned on the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
![alt](../assests/transformer.png)
5) Two independant projection layeres, each layer map the output vector from decoder to task vocabulary Arabic & English. ![alt](../assests/out%20layers.png)

**All In one architecture:**
![alt](../assests/model.png)
---

### Mission

Because of resources limitaion we will simplify the mentioned architecture to be more lighter, scalable and accurate to deploy on production.

We have Advantage of create each component of the model dependent from the others and pretraing it in a supervised and self supervised manner.

