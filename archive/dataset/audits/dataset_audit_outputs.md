# FakeAVCeleb Dataset Audit Report - Cell Outputs
### Cell 3 Output
```text
Dataset Path: d:\uday\Vscode\Projects\Deepfake_Detection\Dataset_FakeAVCeleb
Path exists: True

```
### Cell 6 Output
```text
=== FILE TYPE DISTRIBUTION ===
.mp4: 21,544 files
.txt: 501 files
.csv: 1 files

Total files: 22,046
Total size: 6.16 GB
Average file size: 0.29 MB

```
### Cell 8 Output
```text
=== DETAILED FILE COUNTS BY TYPE ===
Video files: 21,544
Audio files: 0
Image files: 0
Other files: 502

Total dataset size: 6.16 GB
Average file size: 0.29 MB

```
### Cell 10 Output
```text
=== LABEL STORAGE ANALYSIS ===
Found 1 CSV files:
  - d:\uday\Vscode\Projects\Deepfake_Detection\Dataset_FakeAVCeleb\meta_data.csv

Found 0 JSON files:

=== FOLDER-BASED LABELS ===
Top-level folders that might indicate labels:
  - FakeVideo-FakeAudio/
  - FakeVideo-RealAudio/
  - RealVideo-FakeAudio/
  - RealVideo-RealAudio/

```
### Cell 11 Output
```text
=== METADATA CSV ANALYSIS ===
Shape: (21566, 10)

Columns: ['source', 'target1', 'target2', 'method', 'category', 'type', 'race', 'gender', 'path', 'Unnamed: 9']

First 5 rows:
    source target1 target2 method category                 type     race  \
0  id00076       -       -   real        A  RealVideo-RealAudio  African   
1  id00166       -       -   real        A  RealVideo-RealAudio  African   
2  id00173       -       -   real        A  RealVideo-RealAudio  African   
3  id00366       -       -   real        A  RealVideo-RealAudio  African   
4  id00391       -       -   real        A  RealVideo-RealAudio  African   

  gender       path                                         Unnamed: 9  
0    men  00109.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
1    men  00010.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
2    men  00118.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
3    men  00118.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
4    men  00052.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  

Data types:
source        object
target1       object
target2       object
method        object
category      object
type          object
race          object
gender        object
path          object
Unnamed: 9    object
dtype: object

Basic statistics:
         source target1 target2   method category                 type  \
count     21566   21566   21566    21566    21566                21566   
unique      500     526     513        7        4                    4   
top     id00076       -       -  wav2lip        D  FakeVideo-FakeAudio   
freq         88     526   16167     9602    10857                10857   

                        race gender       path  \
count                  21566  21566      21566   
unique                     5      2      18395   
top     Caucasian (American)    men  00028.mp4   
freq                    4864  11107         31   

                                               Unnamed: 9  
count                                               21566  
unique                                               1999  
top     FakeAVCeleb/FakeVideo-FakeAudio/African/men/id...  
freq                                                   55  

```
### Cell 13 Output
```text
=== CLASS DISTRIBUTION FROM FOLDER STRUCTURE ===
FakeVideo-FakeAudio: 10,835 files
FakeVideo-RealAudio: 9,709 files
RealVideo-FakeAudio: 500 files
RealVideo-RealAudio: 1,000 files

=== REAL VS FAKE SUMMARY ===
Total Real samples: 1,500
Total Fake samples: 20,544
Real-to-Fake ratio: 1500:20544 (0.07)
Total samples: 22,044

```
### Cell 15 Output
```text
=== VIDEO ANALYSIS ===
Total video files found: 21,544
Analyzing video 1/100...
Analyzing video 11/100...
Analyzing video 21/100...
Analyzing video 31/100...
Analyzing video 41/100...
Analyzing video 51/100...
Analyzing video 61/100...
Analyzing video 71/100...
Analyzing video 81/100...
Analyzing video 91/100...

Successfully analyzed 100 videos
Average duration: 5.48 seconds
Min duration: 2.04 seconds
Max duration: 17.68 seconds

Unique resolutions:
  224x224: 98 videos
  576x464: 1 videos
  514x482: 1 videos

Average resolution: 230x229

Average FPS: 25.04
FPS range: 23.98 - 29.97

Video formats:
  .mp4: 21544 videos

```
### Cell 17 Output
```text
=== AUDIO ANALYSIS ===
Total audio files found: 0
No audio files could be analyzed

```
### Cell 19 Output
```text
=== MODALITY ANALYSIS ===
Modality Distribution:
  Video+Audio (Fake-Fake): 10,835 files
  Video+Audio (Fake-Real): 9,709 files
  Video+Audio (Real-Fake): 500 files
  Video+Audio (Real-Real): 500 files
  Video Only: 21,544 files
  Audio Only: 0 files
  Images: 0 files

Total multimodal samples (video+audio): 21,544

```
### Cell 21 Output
```text
=== DETAILED METADATA ANALYSIS ===
Found 1 potential metadata files:
  - d:\uday\Vscode\Projects\Deepfake_Detection\Dataset_FakeAVCeleb\meta_data.csv

=== DETAILED ANALYSIS OF meta_data.csv ===

File path: d:\uday\Vscode\Projects\Deepfake_Detection\Dataset_FakeAVCeleb\meta_data.csv
Shape: 21566 rows x 10 columns

Column names and descriptions:
  - source
    Type: object
    Unique values: 500
    Null count: 0

  - target1
    Type: object
    Unique values: 526
    Null count: 0

  - target2
    Type: object
    Unique values: 513
    Null count: 0

  - method
    Type: object
    Unique values: 7
    Values: ['real' 'rtvc' 'faceswap' 'faceswap-wav2lip' 'fsgan' 'fsgan-wav2lip'
 'wav2lip']
    Null count: 0

  - category
    Type: object
    Unique values: 4
    Values: ['A' 'B' 'C' 'D']
    Null count: 0

  - type
    Type: object
    Unique values: 4
    Values: ['RealVideo-RealAudio' 'RealVideo-FakeAudio' 'FakeVideo-RealAudio'
 'FakeVideo-FakeAudio']
    Null count: 0

  - race
    Type: object
    Unique values: 5
    Values: ['African' 'Caucasian (American)' 'Asian (East)' 'Caucasian (European)'
 'Asian (South)']
    Null count: 0

  - gender
    Type: object
    Unique values: 2
    Values: ['men' 'women']
    Null count: 0

  - path
    Type: object
    Unique values: 18395
    Null count: 0

  - Unnamed: 9
    Type: object
    Unique values: 1999
    Null count: 0


First 10 rows:
    source target1 target2 method category                 type     race  \
0  id00076       -       -   real        A  RealVideo-RealAudio  African   
1  id00166       -       -   real        A  RealVideo-RealAudio  African   
2  id00173       -       -   real        A  RealVideo-RealAudio  African   
3  id00366       -       -   real        A  RealVideo-RealAudio  African   
4  id00391       -       -   real        A  RealVideo-RealAudio  African   
5  id00475       -       -   real        A  RealVideo-RealAudio  African   
6  id00476       -       -   real        A  RealVideo-RealAudio  African   
7  id00478       -       -   real        A  RealVideo-RealAudio  African   
8  id00518       -       -   real        A  RealVideo-RealAudio  African   
9  id00701       -       -   real        A  RealVideo-RealAudio  African   

  gender       path                                         Unnamed: 9  
0    men  00109.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
1    men  00010.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
2    men  00118.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
3    men  00118.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
4    men  00052.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
5    men  00099.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
6    men  00109.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
7    men  00206.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
8    men  00031.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  
9    men  00092.mp4  FakeAVCeleb/RealVideo-RealAudio/African/men/id...  

Last 10 rows:
        source  target1 target2   method category                 type  \
21556  id07686  id06439       -  wav2lip        D  FakeVideo-FakeAudio   
21557  id07689  id00747       -  wav2lip        D  FakeVideo-FakeAudio   
21558  id07689  id02310       -  wav2lip        D  FakeVideo-FakeAudio   
21559  id07689  id04070       -  wav2lip        D  FakeVideo-FakeAudio   
21560  id07689  id04530       -  wav2lip        D  FakeVideo-FakeAudio   
21561  id07689  id06254       -  wav2lip        D  FakeVideo-FakeAudio   
21562  id07689  id06343       -  wav2lip        D  FakeVideo-FakeAudio   
21563  id07689  id07008       -  wav2lip        D  FakeVideo-FakeAudio   
21564  id07689  id07377       -  wav2lip        D  FakeVideo-FakeAudio   
21565  id07689  id07686       -  wav2lip        D  FakeVideo-FakeAudio   

                race gender                        path  \
21556  Asian (South)  women  00254_id06439_wavtolip.mp4   
21557  Asian (South)  women  00028_id00747_wavtolip.mp4   
21558  Asian (South)  women  00028_id02310_wavtolip.mp4   
21559  Asian (South)  women  00028_id04070_wavtolip.mp4   
21560  Asian (South)  women  00028_id04530_wavtolip.mp4   
21561  Asian (South)  women  00028_id06254_wavtolip.mp4   
21562  Asian (South)  women  00028_id06343_wavtolip.mp4   
21563  Asian (South)  women  00028_id07008_wavtolip.mp4   
21564  Asian (South)  women  00028_id07377_wavtolip.mp4   
21565  Asian (South)  women  00028_id07686_wavtolip.mp4   

                                              Unnamed: 9  
21556  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  
21557  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  
21558  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  
21559  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  
21560  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  
21561  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  
21562  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  
21563  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  
21564  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  
21565  FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/...  

```
### Cell 23 Output
```text
=== DEEPFAKE CATEGORIES ANALYSIS ===
Manipulation types based on folder structure:
  FakeVideo-FakeAudio: 10,835 samples
    Description: Both video and audio are fake
  FakeVideo-RealAudio: 9,709 samples
    Description: Video is fake, audio is real
  RealVideo-FakeAudio: 500 samples
    Description: Video is real, audio is fake
  RealVideo-RealAudio: 1,000 samples
    Description: Both video and audio are real (ground truth)

=== SUB-CATEGORY ANALYSIS ===

FakeVideo-FakeAudio subcategories:
  - African/: 2,035 files
  - Asian (East)/: 1,649 files
  - Asian (South)/: 2,262 files
  - Caucasian (American)/: 2,505 files
  - Caucasian (European)/: 2,384 files

FakeVideo-RealAudio subcategories:
  - African/: 1,854 files
  - Asian (East)/: 1,635 files
  - Asian (South)/: 1,952 files
  - Caucasian (American)/: 2,159 files
  - Caucasian (European)/: 2,109 files

RealVideo-FakeAudio subcategories:
  - African/: 100 files
  - Asian (East)/: 100 files
  - Asian (South)/: 100 files
  - Caucasian (American)/: 100 files
  - Caucasian (European)/: 100 files

RealVideo-RealAudio subcategories:
  - African/: 200 files
  - Asian (East)/: 200 files
  - Asian (South)/: 200 files
  - Caucasian (American)/: 200 files
  - Caucasian (European)/: 200 files

```
### Cell 25 Output
```text
=== TRAIN/TEST SPLIT ANALYSIS ===
No official train/test split folders found

Metadata does not contain explicit train/test split columns

```
### Cell 27 Output
```text
=== SAMPLE INSPECTION ===
Total real samples available: 500
Total fake samples available: 21,044

=== 10 RANDOM REAL SAMPLES ===
1. RealVideo-RealAudio\African\men\id00830\00143.mp4
   Label: Real
   Duration: 12.48s
   Resolution: 224x224
   FPS: 25.00
2. RealVideo-RealAudio\Asian (East)\men\id08652\00006.mp4
   Label: Real
   Duration: 4.16s
   Resolution: 224x224
   FPS: 25.00
3. RealVideo-RealAudio\Asian (East)\women\id06462\00014.mp4
   Label: Real
   Duration: 4.72s
   Resolution: 224x224
   FPS: 25.00
4. RealVideo-RealAudio\African\women\id00829\00271.mp4
   Label: Real
   Duration: 7.80s
   Resolution: 224x224
   FPS: 25.00
5. RealVideo-RealAudio\Asian (East)\men\id06535\00183.mp4
   Label: Real
   Duration: 6.20s
   Resolution: 224x224
   FPS: 25.00
6. RealVideo-RealAudio\Asian (South)\men\id00350\00015.mp4
   Label: Real
   Duration: 3.96s
   Resolution: 224x224
   FPS: 25.00
7. RealVideo-RealAudio\Asian (East)\women\id09125\00098.mp4
   Label: Real
   Duration: 6.84s
   Resolution: 224x224
   FPS: 25.00
8. RealVideo-RealAudio\Caucasian (European)\men\id00368\00078.mp4
   Label: Real
   Duration: 7.64s
   Resolution: 224x224
   FPS: 25.00
9. RealVideo-RealAudio\Asian (East)\men\id04774\00032.mp4
   Label: Real
   Duration: 4.48s
   Resolution: 224x224
   FPS: 25.00
10. RealVideo-RealAudio\Caucasian (European)\men\id00282\00268.mp4
   Label: Real
   Duration: 6.60s
   Resolution: 224x224
   FPS: 25.00

=== 10 RANDOM FAKE SAMPLES ===
1. FakeVideo-FakeAudio\African\men\id00987\00160_id04727_wavtolip.mp4
   Label: Fake (Fake Video + Fake Audio)
   Duration: 3.72s
   Resolution: 224x224
   FPS: 25.00
2. FakeVideo-FakeAudio\Asian (South)\women\id07686\00254_id06437_3BZvvzBhj8s_id00043_wavtolip.mp4
   Label: Fake (Fake Video + Fake Audio)
   Duration: 5.16s
   Resolution: 224x224
   FPS: 25.00
3. FakeVideo-RealAudio\African\women\id04540\00078_id05235_wavtolip.mp4
   Label: Fake (Fake Video + Real Audio)
   Duration: 5.24s
   Resolution: 224x224
   FPS: 25.00
4. FakeVideo-FakeAudio\Caucasian (American)\men\id01048\00160_0_id00696_wavtolip.mp4
   Label: Fake (Fake Video + Fake Audio)
   Duration: 4.48s
   Resolution: 224x224
   FPS: 25.00
5. FakeVideo-FakeAudio\Caucasian (American)\men\id00696\00005_id04034_wavtolip.mp4
   Label: Fake (Fake Video + Fake Audio)
   Duration: 4.92s
   Resolution: 224x224
   FPS: 25.00
6. FakeVideo-FakeAudio\Caucasian (European)\women\id00826\00065_id00735_ivJSkq5hGOQ_id03589_wavtolip.mp4
   Label: Fake (Fake Video + Fake Audio)
   Duration: 5.64s
   Resolution: 224x224
   FPS: 25.00
7. FakeVideo-FakeAudio\African\women\id01838\00126_id05106_ZnnDdjlABM4_id01907_wavtolip.mp4
   Label: Fake (Fake Video + Fake Audio)
   Duration: 4.84s
   Resolution: 224x224
   FPS: 25.00
8. FakeVideo-RealAudio\Caucasian (European)\women\id00491\00122_id01002_eQTviHjyISo.mp4
   Label: Fake (Fake Video + Real Audio)
   Duration: 4.60s
   Resolution: 224x224
   FPS: 25.00
9. FakeVideo-FakeAudio\African\men\id02296\00019_id00366_wavtolip.mp4
   Label: Fake (Fake Video + Fake Audio)
   Duration: 5.00s
   Resolution: 224x224
   FPS: 25.00
10. FakeVideo-FakeAudio\African\men\id00478\00206_id01544_wavtolip.mp4
   Label: Fake (Fake Video + Fake Audio)
   Duration: 3.92s
   Resolution: 224x224
   FPS: 25.00

```
### Cell 29 Output
```text
=== POTENTIAL ISSUES ANALYSIS ===
1. Checking for empty folders...
   No empty folders found

2. Checking for corrupted files (sample of 50 files)...
   No corrupted files detected in sample

3. Checking for duplicate file names...
   Found 2243 duplicate file names:
     - 00109_id00391_wavtolip.mp4: 2 occurrences
     - 00109_id00475_wavtolip.mp4: 2 occurrences
     - 00109_id00701_wavtolip.mp4: 2 occurrences
     - 00109_id01637_wavtolip.mp4: 2 occurrences
     - 00109_id02005_wavtolip.mp4: 3 occurrences
     - 00109_id02316_wavtolip.mp4: 4 occurrences
     - 00010_id00366_wavtolip.mp4: 4 occurrences
     - 00010_id01076_wavtolip.mp4: 3 occurrences
     - 00010_id01179_wavtolip.mp4: 2 occurrences
     - 00010_id01392_wavtolip.mp4: 4 occurrences

4. Checking for class imbalance...
   Real-to-Fake ratio: 0.07
   WARNING: Significant class imbalance detected!

5. Checking for missing files referenced in metadata...
   Found path columns: ['path', 'Unnamed: 9']
   (Detailed missing file check requires manual inspection of path columns)

```
### Cell 31 Output
```text
================================================================================
FAKEAVCELEB DATASET AUDIT REPORT
================================================================================
Date: June 22, 2026
Dataset Path: d:\uday\Vscode\Projects\Deepfake_Detection\Dataset_FakeAVCeleb
================================================================================

A. EXACT DATASET STRUCTURE
--------------------------------------------------------------------------------
Top-level folders:
  📁 FakeVideo-FakeAudio/
  📁 FakeVideo-RealAudio/
  📁 RealVideo-FakeAudio/
  📁 RealVideo-RealAudio/

Main categories:
  • FakeVideo-FakeAudio: 10,835 files
  • FakeVideo-RealAudio: 9,709 files
  • RealVideo-FakeAudio: 500 files
  • RealVideo-RealAudio: 1,000 files

B. DATASET STATISTICS
--------------------------------------------------------------------------------
Total files: 22,046
Video files: 21,544
Audio files: 0
Image files: 0
Total dataset size: 6.16 GB
Average file size: 0.29 MB

C. LABEL INFORMATION
--------------------------------------------------------------------------------
Labels are stored via:
  • Folder structure (4 main categories)
  • Metadata CSV file (meta_data.csv)
  • Total CSV files: 1
  • Total JSON files: 0

D. CLASS DISTRIBUTION
--------------------------------------------------------------------------------
Real samples: 1,500
Fake samples: 20,544
Real-to-Fake ratio: 1500:20544 (0.07)
Total samples: 22,044

E. VIDEO ANALYSIS
--------------------------------------------------------------------------------
Total videos analyzed: 100
Average duration: 5.48 seconds
Duration range: 2.04 - 17.68 seconds
Average resolution: 230x229
Unique resolutions: 3
Average FPS: 25.04
Video formats: ['.mp4']

F. AUDIO ANALYSIS
--------------------------------------------------------------------------------
Audio analysis data unavailable

G. MODALITY ANALYSIS
--------------------------------------------------------------------------------
  • Video+Audio (Fake-Fake): 10,835
  • Video+Audio (Fake-Real): 9,709
  • Video+Audio (Real-Fake): 500
  • Video+Audio (Real-Real): 500
  • Video Only: 21,544
  • Audio Only: 0
  • Images: 0

Dataset contains: Video + Audio multimodal data

H. DEEPFAKE CATEGORIES
--------------------------------------------------------------------------------
Manipulation types available:
  • FakeVideo-FakeAudio: 10,835 samples
    Both video and audio are fake
  • FakeVideo-RealAudio: 9,709 samples
    Video is fake, audio is real
  • RealVideo-FakeAudio: 500 samples
    Video is real, audio is fake
  • RealVideo-RealAudio: 1,000 samples
    Both video and audio are real (ground truth)

I. TRAIN/TEST SPLIT
--------------------------------------------------------------------------------
No official train/test split folders detected
Recommendation: Create custom splits based on requirements

J. POTENTIAL ISSUES
--------------------------------------------------------------------------------
Empty folders: 0
Corrupted files (sample): 0
Duplicate file names: 2243
Class imbalance: YES (ratio: 0.07)

================================================================================
RECOMMENDATIONS
================================================================================

A. TRAINING STRATEGY
--------------------------------------------------------------------------------
1. For Video-only models:
   • Use all 4 categories
   • Focus on video content analysis
   • Consider frame-level or clip-level processing

2. For Audio-only models:
   • Use FakeVideo-FakeAudio, FakeVideo-RealAudio, RealVideo-FakeAudio
   • Extract audio tracks from videos
   • Focus on spectral analysis

3. For Fusion models (Video + Audio):
   • Ideal for this dataset
   • Leverage multimodal nature
   • Combine video and audio features
   • Use RealVideo-RealAudio as ground truth

B. DATASET SUFFICIENCY
--------------------------------------------------------------------------------
Video model: SUFFICIENT
Audio model: MAY NEED MORE DATA
Fusion model: SUFFICIENT

C. CONCERNS BEFORE TRAINING
--------------------------------------------------------------------------------
1. No official train/test split - need to create custom splits
2. Verify all files are readable before training
3. Check for class imbalance and apply balancing if needed
4. Ensure consistent video/audio quality across categories
5. Validate metadata completeness and accuracy

================================================================================
END OF REPORT

```
### Cell 34 Output
```text
=== AUDIO EXTRACTION AUDIT ===
Testing 100 random videos for audio extraction...

=== 20 RANDOM EXAMPLES ===
1. RealVideo-FakeAudio\Caucasian (European)\men\id00225\00078_fake.mp4 -> Success | 44100Hz | Stereo | 13.52s
2. FakeVideo-FakeAudio\Asian (East)\women\id09174\00015_id02587_KzwJPH5_8j0_faceswap_id06060_wavtolip.mp4 -> Success | 44100Hz | Stereo | 4.14s
3. FakeVideo-FakeAudio\African\men\id01779\00010_id01691_IVtS5z8Jrrk_faceswap_id01779_wavtolip.mp4 -> Success | 44100Hz | Stereo | 5.25s
4. FakeVideo-FakeAudio\Caucasian (European)\men\id00519\00028_id01098_wavtolip.mp4 -> Success | 44100Hz | Stereo | 3.39s
5. FakeVideo-FakeAudio\Caucasian (American)\women\id01005\00028_id01225_wavtolip.mp4 -> Success | 44100Hz | Stereo | 5.60s
6. FakeVideo-FakeAudio\Caucasian (American)\women\id00097\00162_id00180_lkeAa8Od_tg_id00231_wavtolip.mp4 -> Success | 44100Hz | Stereo | 4.42s
7. FakeVideo-FakeAudio\Asian (South)\men\id07179\00206_id07165_oJ_iy16IBTM_faceswap_id07233_wavtolip.mp4 -> Success | 44100Hz | Stereo | 4.04s
8. FakeVideo-FakeAudio\Asian (East)\women\id06066\00028_id00363_wavtolip.mp4 -> Success | 44100Hz | Stereo | 3.65s
9. FakeVideo-RealAudio\Caucasian (American)\women\id00618\00195_id01004_867Wlj7Gw68.mp4 -> Success | 44100Hz | Stereo | 5.57s
10. FakeVideo-FakeAudio\Asian (East)\men\id08613\00074_id00597_wavtolip.mp4 -> Success | 44100Hz | Stereo | 4.98s
11. FakeVideo-RealAudio\Caucasian (European)\men\id01123\00072_id00305_LdcY3ILdQrE.mp4 -> Success | 44100Hz | Stereo | 8.77s
12. FakeVideo-RealAudio\Asian (East)\women\id05620\00005_id06388_wavtolip.mp4 -> Success | 44100Hz | Stereo | 5.70s
13. FakeVideo-FakeAudio\African\men\id02296\00019_id01392_wavtolip.mp4 -> Success | 44100Hz | Stereo | 5.10s
14. FakeVideo-FakeAudio\African\men\id02040\00476_id00476_wavtolip.mp4 -> Success | 44100Hz | Stereo | 4.30s
15. FakeVideo-FakeAudio\Asian (East)\women\id02587\00020_id04701_wavtolip.mp4 -> Success | 44100Hz | Stereo | 4.36s
16. FakeVideo-FakeAudio\Caucasian (American)\men\id04034\00009_id00971_edWrVVn-rC4_faceswap_id03668_wavtolip.mp4 -> Success | 44100Hz | Stereo | 10.56s
17. FakeVideo-FakeAudio\Caucasian (American)\women\id00398\00016_id01231_XXpYdOHUF-g_id01217_wavtolip.mp4 -> Success | 44100Hz | Stereo | 5.78s
18. FakeVideo-RealAudio\Caucasian (American)\men\id00243\00037_id00345_wavtolip.mp4 -> Success | 44100Hz | Stereo | 4.10s
19. FakeVideo-RealAudio\Caucasian (European)\women\id00325\00015_id03858_wavtolip.mp4 -> Success | 44100Hz | Stereo | 6.78s
20. FakeVideo-FakeAudio\African\men\id01920\00099_id00366_RSOgEoek8WM_id00166_wavtolip.mp4 -> Success | 44100Hz | Stereo | 5.04s

=== OVERALL AUDIO EXTRACTION STATISTICS ===
Total Videos Checked: 100
Successfully Extracted: 100
Missing Audio: 0
Extraction Errors: 0

Average Audio Duration: 5.49 seconds
Min Duration: 1.92 seconds
Max Duration: 14.08 seconds

Sample Rate Distribution:
  44100 Hz: 100 videos

Channel Distribution:
  Stereo: 100 videos

```
### Cell 36 Output
```text
=== IDENTITY LEAKAGE AUDIT ===
Total unique source identities: 500
Total unique target identities: 578

Number of source identities appearing in each category (type):
  - RealVideo-RealAudio: 500 identities
  - RealVideo-FakeAudio: 500 identities
  - FakeVideo-RealAudio: 500 identities
  - FakeVideo-FakeAudio: 499 identities

Does the same identity appear in BOTH real and fake samples? YES
  - Found 500 identities appearing in both real and fake sets.
  - Examples: ['id06152', 'id03344', 'id01933', 'id00078', 'id00071']

Does the same source identity appear hundreds of times? 
  - NO, max occurrences is 88.

Top 20 most frequent source identities:
  - id00076: 88 times
  - id01182: 68 times
  - id00520: 67 times
  - id03525: 66 times
  - id00183: 66 times
  - id00264: 66 times
  - id01210: 66 times
  - id00841: 65 times
  - id04073: 65 times
  - id03757: 65 times
  - id00100: 64 times
  - id01048: 64 times
  - id01058: 63 times
  - id00068: 63 times
  - id01075: 63 times
  - id07136: 61 times
  - id04526: 61 times
  - id04537: 60 times
  - id07689: 60 times
  - id00032: 60 times

=== RECOMMENDED SPLITTING STRATEGY ===
RECOMMENDATION: **IDENTITY-BASED SPLIT**
Reasoning: 
  - Identities appear in both real and fake samples.
  - Identities appear multiple times across samples.
  - A random split will cause identity leakage, allowing the model to cheat by memorizing faces/voices.

Suggested Identity Split (70/15/15):
  - Train identities: 350 (70.0%)
    Examples: ['id04374', 'id00345', 'id04561', 'id00752', 'id00462']...
  - Validation identities: 75 (15.0%)
    Examples: ['id01530', 'id05844', 'id00029', 'id01544', 'id06753']...
  - Test identities: 75 (15.0%)
    Examples: ['id00368', 'id01058', 'id00171', 'id01683', 'id06268']...

```
### Cell 38 Output
```text
=== LABEL INTEGRITY AUDIT ===
Checking 500 random samples for label integrity...

Total Mismatches Found: 24

Examples of mismatches:
  - FakeVideo-FakeAudio\Asian (South)\women\id05434\00052_id00488_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\African\men\id00478\00206_id01544_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\African\men\id01598\00044_id00987_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Asian (South)\women\id06428\00043_id05845_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\African\men\id01972\00078_id00781_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Caucasian (American)\women\id00100\00028_id00180_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Caucasian (European)\men\id00225\00078_id00358_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\African\men\id01452\00001_id00478_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Asian (South)\men\id06334\00015_id07210_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Asian (South)\men\id06753\00021_id06355_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio

```
### Cell 40 Output
```text
=== IDENTITY SPLIT STATISTICS ===

Train Split (350 identities):
  - Total Videos: 15099
  - Real Samples: 350
  - Fake Samples: 14749
  - Real-to-Fake Ratio: 0.0237

Validation Split (75 identities):
  - Total Videos: 3194
  - Real Samples: 75
  - Fake Samples: 3119
  - Real-to-Fake Ratio: 0.0240

Test Split (75 identities):
  - Total Videos: 3273
  - Real Samples: 75
  - Fake Samples: 3198
  - Real-to-Fake Ratio: 0.0235

Check completed. Ensuring test set is appropriately balanced.

```
### Cell 42 Output
```text
=== LABEL INTEGRITY AUDIT ===
Checking 500 random samples for label integrity...

Total Mismatches Found: 24

Examples of mismatches:
  - FakeVideo-FakeAudio\Asian (South)\women\id05434\00052_id00488_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\African\men\id00478\00206_id01544_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\African\men\id01598\00044_id00987_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Asian (South)\women\id06428\00043_id05845_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\African\men\id01972\00078_id00781_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Caucasian (American)\women\id00100\00028_id00180_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Caucasian (European)\men\id00225\00078_id00358_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\African\men\id01452\00001_id00478_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Asian (South)\men\id06334\00015_id07210_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio
  - FakeVideo-FakeAudio\Asian (South)\men\id06753\00021_id06355_wavtolip.mp4 -> Folder: FakeVideo-FakeAudio, Meta: FakeVideo-RealAudio

```
### Cell 44 Output
```text
=== IDENTITY SPLIT STATISTICS ===

Train Split (350 identities):
  - Total Videos: 15099
  - Real Samples: 350
  - Fake Samples: 14749
  - Real-to-Fake Ratio: 0.0237

Validation Split (75 identities):
  - Total Videos: 3194
  - Real Samples: 75
  - Fake Samples: 3119
  - Real-to-Fake Ratio: 0.0240

Test Split (75 identities):
  - Total Videos: 3273
  - Real Samples: 75
  - Fake Samples: 3198
  - Real-to-Fake Ratio: 0.0235

Check completed. Ensuring test set is appropriately balanced.

```
