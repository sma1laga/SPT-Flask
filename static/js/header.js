// Wait for the page to load
document.addEventListener("DOMContentLoaded", function() {
  const darkModeToggle = document.getElementById("dark-mode-toggle");
  const colorblindToggle = document.getElementById("colorblind-mode-toggle");
  if (darkModeToggle) {
    darkModeToggle.addEventListener("click", function() {
      document.body.classList.toggle("dark-mode");
      // Save the preference in local storage
      if (document.body.classList.contains("dark-mode")) {
        localStorage.setItem("darkMode", "true");
        darkModeToggle.textContent = "Light Mode";
      } else {
        localStorage.setItem("darkMode", "false");
        darkModeToggle.textContent = "Dark Mode";
      }
    });
  
    // On page load, check local storage for dark mode preference
    if (localStorage.getItem("darkMode") === "true") {
      document.body.classList.add("dark-mode");
      darkModeToggle.textContent = "Light Mode";
    }
  }

  if (colorblindToggle) {
    const toggleContainer = colorblindToggle.closest(".toggle-container");

    const updateColorblindUI = (isEnabled) => {
      document.body.classList.toggle("colorblind-mode", isEnabled);


      if (toggleContainer) {
        toggleContainer.classList.toggle("is-active", isEnabled);
      }
    };

    colorblindToggle.addEventListener("change", function() {
      const isEnabled = colorblindToggle.checked;
      updateColorblindUI(isEnabled);
      localStorage.setItem("colorblindMode", isEnabled ? "true" : "false");
    });

    if (localStorage.getItem("colorblindMode") === "true") {
      colorblindToggle.checked = true;
    }
    updateColorblindUI(colorblindToggle.checked);
  }

  // Sidebar toggle for mobile
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebar = document.querySelector(".sidebar");
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", function() {
      sidebar.classList.toggle("active");
    });
  }
});
