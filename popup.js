document.getElementById("scanButton").addEventListener("click", async () => {
  document.getElementById("report").textContent = "🔍 Scanning images on this page...";

  // Send message to content script
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.tabs.sendMessage(tab.id, { action: "scan_page" });

  setTimeout(() => {
    document.getElementById("report").textContent = "✅ Scan complete! Check labels on images.";
  }, 4000);
});
