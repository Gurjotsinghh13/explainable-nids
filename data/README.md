# Data

The **CICIDS2017 (MachineLearningCVE)** dataset is NOT stored in this repo.

Your extracted copy currently lives at:

```
D:\Capstone Project\MachineLearningCVE
```

This is configured as the default `DATA_DIR` in `src/config.py`. You can either:

- Leave the CSVs where they are and rely on that default path, **or**
- Copy the 8 CSVs into this `data/` folder and point `DATA_DIR` here.

## Expected files (8 CSVs)

| File                                             | Attack content                          |
| ------------------------------------------------ | --------------------------------------- |
| Monday-WorkingHours.pcap_ISCX.csv                | Benign only (baseline)                  |
| Tuesday-WorkingHours.pcap_ISCX.csv               | Brute Force (FTP-Patator, SSH-Patator)  |
| Wednesday-workingHours.pcap_ISCX.csv             | DoS variants + Heartbleed               |
| Thursday-WorkingHours-Morning-WebAttacks...csv   | Web Attacks (Brute Force, XSS, SQLi)    |
| Thursday-WorkingHours-Afternoon-Infilteration... | Infiltration                            |
| Friday-WorkingHours-Morning.pcap_ISCX.csv        | Botnet                                  |
| Friday-WorkingHours-Afternoon-PortScan...csv     | PortScan                                |
| Friday-WorkingHours-Afternoon-DDos...csv         | DDoS                                    |

## Known quirk (Web Attacks file)

The Thursday Web Attacks CSV contains a non-UTF-8 byte in some labels
(e.g. `Web Attack ? Brute Force`). We handle this during loading/preprocessing
— just be aware of it if you open that file manually.
