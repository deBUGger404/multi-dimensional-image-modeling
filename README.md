# multi-dimension-modeling
## Dataset
model build for multi dimension images like image dimension >3(rgb). so I bulid random 8 dimension images and there respective random labels.
```
        self.image = np.random.rand(5000,224,224,8)
        self.labels = np.random.choice([0, 1], size=(5000,), p=[0.6,0.4])
```
