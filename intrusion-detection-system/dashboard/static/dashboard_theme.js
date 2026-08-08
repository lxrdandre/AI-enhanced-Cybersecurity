(function () {
  const STORAGE_KEY = "toniot_dashboard_theme";

  function readTheme() {
    try {
      return localStorage.getItem(STORAGE_KEY) === "light" ? "light" : "dark";
    } catch (_err) {
      return "dark";
    }
  }

  function saveTheme(theme) {
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch (_err) {
      // The button still works for the current page if storage is blocked.
    }
  }

  function applyTheme(theme) {
    const isLight = theme === "light";
    document.documentElement.dataset.theme = isLight ? "light" : "dark";

    const button = document.querySelector("[data-theme-toggle]");
    if (!button) return;

    button.setAttribute("aria-pressed", String(isLight));
    button.setAttribute("aria-label", isLight ? "Switch to dark theme" : "Switch to light theme");
    button.title = isLight ? "Switch to dark theme" : "Switch to light theme";

    const label = button.querySelector("[data-theme-toggle-label]");
    if (label) label.textContent = isLight ? "Dark theme" : "Light theme";
  }

  function initThemeToggle() {
    applyTheme(readTheme());

    const button = document.querySelector("[data-theme-toggle]");
    if (!button) return;

    button.addEventListener("click", () => {
      const nextTheme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      applyTheme(nextTheme);
      saveTheme(nextTheme);
    });
  }

  applyTheme(readTheme());

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initThemeToggle);
  } else {
    initThemeToggle();
  }
}());
