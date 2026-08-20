# DENOISING U-NET

**Brusreducering av bilder med en U-Net autoencoder**

Projektet tränar en U-Net-modell med skip connections för att ta bort Gaussiskt brus från bilder. Modellen får en brusig bild som input och lär sig att rekonstruera motsvarande rena bild.

## Dataflöde

1. JPG- och JPEG-bilder läses in från `dataset/`.
2. Originalbilderna blandas reproducerbart och delas upp i train, val och test enligt procentsatserna i `config.py`.
3. Uppdelningen sker innan tiles skapas. Tiles från samma originalbild kan därför inte hamna i olika splits.
4. Varje bild centreras och delas upp i kompletta tiles på 256×256 pixlar. Bilddelar som inte fyller en hel tile används inte.
5. En enkel EDA körs före träningen och sparas i `outputs/eda/`.
6. Modellen tränas endast på tiles från train-splitten. Varje tile får en egen slumpad brusnivå mellan `MIN_NOISE` och `MAX_NOISE`, och den rena tilen används som target.
7. Validation loss beräknas på tiles från val-splitten med den fasta brusnivån `VAL_NOISE`. Samma brusnivå används även för båda testerna.
8. Efter träningen testas modellen på tiles från test-splitten.
9. Modellen testas även på hela originalbilder från test-splitten. Längsta sidan skalas ned till maximalt 256 pixlar, proportionerna bevaras och bilden paddas till 256×256.

> Modellen tränas inte på hela bilder. Helbilderna används endast efter träningen för att undersöka hur modellen generaliserar utanför tile-formatet.

## Modell

Modellen är en deterministisk denoising autoencoder med U-Net-struktur. Encodern minskar den spatiala storleken, decodern skalar upp bilden igen och skip connections skickar detaljer från encodern direkt till motsvarande nivå i decodern.

Modellen optimeras med Adam och MSE loss. EarlyStopping använder validation loss och den bästa modellen sparas automatiskt.

## Utvärdering

En classification report och confusion matrix används inte eftersom modellen rekonstruerar kontinuerliga pixelvärden i stället för att förutsäga klasser. Efter träningen jämförs den brusiga baselinen och modellens rekonstruktion med följande mått:

- MSE och MAE, där lägre värden är bättre.
- PSNR och SSIM, där högre värden är bättre.

Måtten beräknas separat för test-tiles och hela testbilder. För helbilderna räknas endast den verkliga bildytan, inte paddingen.

## Outputs

- `outputs/eda/` innehåller EDA-figurer och en textsammanfattning.
- `outputs/checkpoint/` innehåller den bästa sparade modellen.
- `outputs/results/training_history.png` visar training och validation loss.
- `outputs/results/training_history.csv` innehåller loss per epoch.
- `outputs/results/evaluation_summary.txt` innehåller den samlade utvärderingsrapporten.
- `outputs/results/tile_metrics.csv` innehåller mått per test-tile.
- `outputs/results/whole_image_metrics.csv` innehåller mått per testbild.
- `outputs/results/metric_comparison.png` jämför baseline och rekonstruktion.
- `outputs/results/reconstructions.png` visar testresultat för tiles.
- `outputs/results/whole_image_reconstructions.png` visar testresultat för hela bilder.

## Kör projektet

Lägg bilderna i `dataset/` och kör:

```bash
python app.py
```

Träningsinställningar, datasetandelar och output-sökvägar finns i `config.py`.
