# TWO-STAGE DENOISING U-NET

**Brusreducering och efterföljande residualförfining med två U-Net-modeller**

Projektet tränar två U-Net-modeller med skip connections. Steg ett tar bort Gaussiskt brus. Steg två får rekonstruktionerna från steg ett som input och lär sig en residual, alltså en korrigering som adderas till bilden. Båda modellerna använder motsvarande rena bild som target.

## Dataflöde

1. JPG- och JPEG-bilder läses in från `dataset/`.
2. Originalbilderna blandas reproducerbart och delas upp i train, val och test enligt procentsatserna i `config.py`.
3. Uppdelningen sker innan tiles skapas. Tiles från samma originalbild kan därför inte hamna i olika splits.
4. Varje bild centreras och delas upp i kompletta tiles på 256×256 pixlar. Bilddelar som inte fyller en hel tile används inte.
5. En enkel EDA körs före träningen och sparas i `outputs/eda/`.
6. Steg ett tränas endast på train-tiles. Varje tile får en egen slumpad Gaussisk brusnivå mellan `MIN_NOISE` och `MAX_NOISE`.
7. Validation loss för steg ett beräknas med den fasta brusnivån `VAL_NOISE`.
8. Den bästa modellen från steg ett skapar en rekonstruktion för varje train- och val-tile.
9. Rekonstruktionerna och deras rena targets sparas som parade PNG-filer i `dataset_stage_two/`.
10. Steg två tränas från början på `rekonstruktion från steg ett → ren tile`. Modellen förutsäger en residual som adderas till rekonstruktionen. Inget extra brus läggs till.
11. Efter båda träningsstegen körs det orörda testsetet genom hela kedjan: `brusig → steg ett → steg två`.
12. Kedjan testas både på test-tiles och på hela testbilder som skalats proportionellt till maximalt 256 pixlar och paddats till 256×256.

> Modellerna tränas inte på hela bilder. Helbilderna används endast efter träningen för att undersöka hur modellerna generaliserar utanför tile-formatet.

## Genererat dataset för steg två

`dataset_stage_two/` byggs om efter varje slutförd träning av steg ett:

```text
dataset_stage_two/
├── train/
│   ├── inputs/
│   └── targets/
├── val/
│   ├── inputs/
│   └── targets/
└── manifest.csv
```

Input och target har samma filnamn. Manifestet innehåller split, sökvägar och brusnivån som användes före rekonstruktionen. Testdata sparas aldrig i detta dataset.

## Modeller

Båda stegen använder samma U-Net-backbone men skilda vikter och initialiserings-seeds. Varje encoder-, bottleneck- och decoder-nivå använder två efterföljande 3×3-convolutioner. Encodern minskar den spatiala storleken, decodern skalar upp bilden igen och skip connections skickar detaljer från encodern till motsvarande nivå i decodern.

Steg ett skapar en komplett denoisad bild. Steg två har i stället ett residualhuvud som startar med nollvikter. Modellen börjar därför som en identitetsfunktion och lär sig endast den korrigering som behövs. Den korrigerade bilden begränsas till intervallet 0–1.

Modellerna optimeras separat med Adam och MSE loss. Varje steg har egen EarlyStopping och egen checkpoint. Modellen från steg ett tas bort från GPU-minnet innan steg två byggs.

## Utvärdering

En classification report och confusion matrix används inte eftersom modellerna rekonstruerar kontinuerliga pixelvärden i stället för att förutsäga klasser. Följande tre versioner jämförs:

- Brusig baseline.
- Rekonstruktion efter steg ett.
- Rekonstruktion efter steg två.

MSE och MAE ska minska medan PSNR och SSIM ska öka. Måtten beräknas separat för test-tiles och hela testbilder. För helbilderna räknas endast den verkliga bildytan, inte paddingen. Rapporten visar uttryckligen om steg två förbättrar eller försämrar resultatet jämfört med steg ett.

## Outputs

- `outputs/eda/` innehåller EDA-figurer och en textsammanfattning.
- `outputs/checkpoint/best_stage_one_unet.keras` innehåller bästa modellen från steg ett.
- `outputs/checkpoint/best_stage_two_residual_unet.keras` innehåller bästa residualmodellen från steg två.
- `outputs/results/stage_one_training_history.*` innehåller historik för steg ett.
- `outputs/results/stage_two_training_history.*` innehåller historik för steg två.
- `outputs/results/evaluation_summary.txt` innehåller den samlade utvärderingsrapporten.
- `outputs/results/tile_metrics.csv` innehåller mått per test-tile.
- `outputs/results/whole_image_metrics.csv` innehåller mått per testbild.
- `outputs/results/metric_comparison.png` jämför baseline och båda modellerna.
- `outputs/results/two_stage_tile_reconstructions.png` visar testresultat för tiles.
- `outputs/results/two_stage_whole_image_reconstructions.png` visar testresultat för hela bilder.

## Kör projektet

Lägg bilderna i `dataset/` och kör:

```bash
python app.py
```

Båda träningsstegen körs automatiskt efter varandra. Träningsinställningar, datasetandelar och output-sökvägar finns i `config.py`.
