chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((error) => console.error(error));

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: 'analyzeImage',
        title: 'Authentic.AI: Analyze this image',
        contexts: ['image']
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'analyzeImage') {
        // In a real implementation, we would send this URL to the sidepanel
        // For now, we just open the panel
        chrome.sidePanel.open({ windowId: tab.windowId });
    }
});
