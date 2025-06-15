async function fetchAndSendCountry() {
  try {
    let country = localStorage.getItem('country');
    if (!country) {
      const res = await fetch('https://ipapi.co/json/');
      if (res.ok) {
        const data = await res.json();
        country = data.country_name;
        if (country) {
          localStorage.setItem('country', country);
        }
      }
    }
    if (country) {
      fetch('/analytics/country', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ country })
      });
    }
  } catch (e) {
    console.error('Country fetch failed', e);
  }
}

document.addEventListener('DOMContentLoaded', fetchAndSendCountry);