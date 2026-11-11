# Five-minute Stego Lab demo

## Start the app

From the project root in PowerShell, run:

```powershell
.\.venv\Scripts\python.exe app.py
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
8. **Run detection (about 45 seconds).** Click **04 / Detect**, upload either stego PNG, then click **Analyze image**. Compare the LSB chi-square, sequential LSB, and DCT parity verdicts. Point out that DCT parity is configured for quality 75 only and that the page lists detector limitations.
