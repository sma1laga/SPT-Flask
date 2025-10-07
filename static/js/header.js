// Wait for the page to load
document.addEventListener("DOMContentLoaded", function() {
  const darkModeToggle = document.getElementById("dark-mode-toggle");
  const colorblindToggle = document.getElementById("colorblind-mode-toggle");
  const cookieMaxAge = 60 * 60 * 24 * 365; // one year

  const getCookie = (name) => {
    const cookieString = `; ${document.cookie}`;
    const parts = cookieString.split(`; ${name}=`);
    if (parts.length === 2) {
      return parts.pop().split(";").shift();
    }
    return null;
  };

  const applyDarkModePreference = (isEnabled) => {
    document.body.classList.toggle("dark-mode", isEnabled);
    localStorage.setItem("darkMode", isEnabled ? "true" : "false");
    document.cookie = `darkMode=${isEnabled ? "true" : "false"}; path=/; max-age=${cookieMaxAge}; SameSite=Lax`;
    if (darkModeToggle) {
      darkModeToggle.textContent = isEnabled ? "Light Mode" : "Dark Mode";
    }
  };

  let isDarkMode = document.body.classList.contains("dark-mode");
  const storedPreference = localStorage.getItem("darkMode");
  if (storedPreference !== null) {
    isDarkMode = storedPreference === "true";
  } else {
    const cookiePreference = getCookie("darkMode");
    if (cookiePreference !== null) {
      isDarkMode = cookiePreference === "true";
    }
  }

  applyDarkModePreference(isDarkMode);
  if (darkModeToggle) {
    darkModeToggle.addEventListener("click", function() {
      applyDarkModePreference(!document.body.classList.contains("dark-mode"));

    });

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

  // Mobile navigation toggle
  const navToggle = document.querySelector(".nav-toggle");
  const mainNav = document.querySelector(".main-nav");
  const navBackdrop = document.querySelector(".nav-backdrop");

  if (navToggle && mainNav) {
    const setNavState = (isOpen) => {
      mainNav.classList.toggle("is-open", isOpen);
      navToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
      document.body.classList.toggle("nav-open", isOpen);
      if (navBackdrop) {
        navBackdrop.classList.toggle("is-visible", isOpen);
      }
    };

    navToggle.addEventListener("click", (event) => {
      event.stopPropagation();
      const isOpen = !mainNav.classList.contains("is-open");
      setNavState(isOpen);
    });

    if (navBackdrop) {
      navBackdrop.addEventListener("click", () => setNavState(false));
    }

    mainNav.addEventListener("click", (event) => {
      if (window.innerWidth <= 900 && event.target.closest("a, button")) {
        setNavState(false);
      }
    });

    document.addEventListener("click", (event) => {
      if (
        window.innerWidth <= 900 &&
        mainNav.classList.contains("is-open") &&
        !mainNav.contains(event.target) &&
        !navToggle.contains(event.target)
      ) {
        setNavState(false);
      }
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth > 900 && mainNav.classList.contains("is-open")) {
        setNavState(false);
      }
    });
  }
});
