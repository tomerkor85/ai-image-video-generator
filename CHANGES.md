# שינויים שבוצעו בפרויקט

## תמיכה במודלי LORA

### FLUX.1-dev עם LORA
- ✅ תמיכה מלאה ב-LORA עבור FLUX.1-dev
- ✅ קובץ LORA: `models/flux_naya.safetensors`
- ✅ טעינה אוטומטית של LORA עם המודל
- ✅ שליטה ב-LORA scale (0.0 - 2.0)

### WAN 2.2 עם LORA
- ✅ תמיכה מלאה ב-LORA עבור WAN 2.2 (Stable Video Diffusion)
- ✅ שני סוגי LORA: High Noise ו-Low Noise
- ✅ בחירה בין סוגי LORA ב-UI
- ✅ טעינה אוטומטית של LORA עם המודל
- ✅ שליטה ב-LORA scale

## API חדשים

### `/generate/video` - ייצור וידאו
```json
{
  "prompt": "a beautiful sunset over mountains",
  "negative_prompt": "blurry, low quality",
  "width": 512,
  "height": 512,
  "num_frames": 16,
  "steps": 25,
  "guidance": 7.5,
  "seed": null,
  "lora_scale": 1.0,
  "lora_type": "high"
}
```

### `/outputs/videos/{filename}` - קבלת קבצי וידאו
- מחזיר קבצי MP4 שנוצרו

## שיפורי UI

### בחירת סוג תוכן
- 🖼️ ייצור תמונות
- 🎬 ייצור וידאו

### בחירת מודל
- 🚀 FLUX.1-dev (עם LORA)
- 🎯 Stable Diffusion (ללא LORA)

### אפשרויות וידאו
- מספר פריימים (8, 16, 24, 32)
- בחירת סוג LORA (High Noise / Low Noise)
- שליטה ב-LORA scale
- תצוגת וידאו בגלריה

### גלריה משופרת
- הצגת תמונות ווידאו יחד
- תגיות IMG/VID
- תצוגה מקדימה של וידאו על hover

## מבנה קבצים

```
ai-gen/
├── main.py                 # שרת FastAPI מעודכן
├── flux_generator.py       # מחלקת FLUX עם LORA
├── wan_generator.py        # מחלקת WAN עם LORA
├── config.py              # הגדרות מעודכנות
├── models/
│   └── flux_naya.safetensors  # קובץ LORA ל-FLUX
├── naya_wan_lora/
│   ├── lora_t2v_A14B_separate_high.safetensors  # LORA רעש גבוה
│   └── lora_t2v_A14B_separate_low.safetensors   # LORA רעש נמוך
├── outputs/
│   ├── images/            # תמונות שנוצרו
│   └── videos/            # וידאו שנוצר
└── web_ui.html            # UI מעודכן
```

## שימוש

### הפעלת השרת
```bash
python main.py
```

### גישה ל-UI
- http://localhost:8888/ui

### API Documentation
- http://localhost:8888/docs

## תכונות חדשות

1. **תמיכה מלאה ב-LORA** - גם ב-FLUX וגם ב-WAN
2. **שני סוגי LORA ל-WAN** - High Noise ו-Low Noise
3. **ייצור וידאו** - עם WAN 2.2
4. **UI משופר** - בחירת סוג תוכן ומודל
5. **בחירת סוג LORA** - High/Low Noise לווידאו
6. **גלריה משולבת** - תמונות ווידאו יחד
7. **API מלא** - endpoints לכל הפונקציונליות

## הערות טכניות

- LORA נטען אוטומטית עם המודל
- תמיכה ב-LORA scale מ-0.0 עד 2.0
- בחירה בין High Noise ו-Low Noise LORA לווידאו
- וידאו נוצר ב-MP4 עם 8 FPS
- זיכרון GPU מנוהל אוטומטית
- תמיכה ב-CUDA ו-CPU
- Fallback אוטומטי אם LORA מסוים לא קיים
