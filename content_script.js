async function analyzeImages() {
  const images = document.querySelectorAll("img");
  console.log(`Found ${images.length} images`);

  for (const img of images) {
    try {
      // Skip small icons, logos, or base64 images
      if (!img.src || img.src.startsWith("data:") || img.width < 50 || img.height < 50) continue;

      // Fetch image as blob
      const response = await fetch(img.src, { mode: "cors" });
      const blob = await response.blob();

      // Create form data
      const formData = new FormData();
      formData.append("file", blob, "image.jpg");

      // Send to FastAPI backend
      const res = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      console.log("Prediction result:", data);

      // Create overlay label
      const label = document.createElement("div");
      label.textContent = `${data.prediction} (${data.confidence.toFixed(2)}%)`;
      label.style.position = "absolute";
      label.style.top = "5px";
      label.style.right = "5px";
      label.style.backgroundColor =
        data.prediction === "Real"
          ? "rgba(0, 200, 0, 0.8)"
          : "rgba(255, 0, 0, 0.8)";
      label.style.color = "white";
      label.style.padding = "3px 6px";
      label.style.borderRadius = "8px";
      label.style.fontSize = "12px";
      label.style.zIndex = "9999";
      label.style.fontFamily = "Poppins, sans-serif";

      // Wrap image in relative container
      const wrapper = document.createElement("div");
      wrapper.style.position = "relative";
      wrapper.style.display = "inline-block";
      img.parentNode.insertBefore(wrapper, img);
      wrapper.appendChild(img);
      wrapper.appendChild(label);
    } catch (error) {
      console.error("Error analyzing image:", error);
    }
  }
}

// Listen for signal from popup
chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "scan_page") {
    analyzeImages();
  }
});
