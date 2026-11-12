# Five-minute Stego Lab demo

## Start the app

From the project root in PowerShell, run:

```powershell
.\venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/). The app is a local demo and is not for public deployment.

## Click-through script

1. **Hide with shuffled LSB (about 45 seconds).** Click **02 / Hide** in the top navigation. Click **LSB password-shuffled** in the method cards. Choose a PNG or JPG cover, enter `Meet me at the library at noon.`, enter `mentor-demo-42` as the password, then click **Hide message**. Point out the UTF-8 byte count and maximum message size. Click **Download stego PNG**.
2. **Extract without the password (about 30 seconds).** Click **03 / Extract**. Click **LSB password-shuffled**. Choose the downloaded `stego.png`, leave Password blank, then click **Extract message**. Show **NO MESSAGE FOUND**.
3. **Extract with the password (about 30 seconds).** Keep the same method and image, enter `mentor-demo-42`, and click **Extract message**. Show **INTACT** and the original message.
4. **Show JPEG damage to LSB (about 40 seconds).** Keep the shuffled LSB image and password selected. Check **Simulate JPEG compression first**, leave quality at **70**, and click **Extract message**. Show that the message is no longer recovered.
5. **Hide robust DCT (about 45 seconds).** Click **02 / Hide**, select **DCT robust (quality 50)**, upload the original cover, enter the same message and optional password, and click **Hide message**. Download the new PNG.
6. **Show DCT surviving JPEG 70 (about 40 seconds).** Click **03 / Extract**, select **DCT robust (quality 50)**, upload the DCT PNG, enter the optional password, check JPEG simulation at quality **70**, then click **Extract message**. Show **INTACT** and explain that extraction must use the same embedding quality. The DCT password authenticates but does not conceal the message.
7. **Demonstrate tampering (about 35 seconds).** Return to **03 / Extract**, select the shuffled LSB method, upload its downloaded PNG, enter the password, check **Flip random pixel LSBs**, set the count to **50000**, and click **Extract message**. Show the text status and the number of edited pixels.
8. **Check a short message (about 45 seconds).** Click **02 / Hide**, select **LSB plain**, upload the original cover, and enter `Meet me at noon.`. Click **Hide message**, download the PNG, then click **04 / Detect**, upload that PNG, and click **Analyze image**. Show **no evidence** for the short payload; this is expected because these detectors miss short messages.
9. **Show sequential detection at 10% (about 45 seconds).** Click **02 / Hide**, select **LSB plain**, upload the original cover, select **10% of capacity** under **Detector demo**, then click **Build plain LSB demo**. The message field is ignored. Download the demonstration PNG and click **04 / Detect**. The sequential test should say **embedding likely** and show its estimated payload end.
10. **Show whole-image detection on shuffled LSB (about 45 seconds).** Click **02 / Hide**, select **LSB password-shuffled**, upload the original cover, and click **Build shuffled LSB demo at 100%**. The message and password fields are ignored for this seeded detector demonstration. Download the PNG and run **04 / Detect**; the whole-image LSB test should say **embedding likely**.

The DCT detector (quality 75 only) fires on 22/23 at 100% and falsely flags 2 of 23 (about 9%) of clean full covers and 12% (512x384 crops) to 22% (256x192 crops) of small crops.
