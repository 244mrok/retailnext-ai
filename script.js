// ユーザー名を保持する変数
let currentUser = null;

document.addEventListener('DOMContentLoaded', function () {
  showLoading(false); // 画面ロード直後にローディング非表示

  // DOM要素の取得
  const header = document.getElementById('header');
  const hamburgerBtn = document.getElementById('hamburger-btn');
  const mobileMenu = document.getElementById('mobile-menu');

  // スクロール時のヘッダー背景変更
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  });

  // ハンバーガーメニューの開閉
  hamburgerBtn.addEventListener('click', () => {
    mobileMenu.classList.toggle('hidden');

    // アイコンをメニューから閉じるへ切り替え
    const icon = hamburgerBtn.querySelector('i');
    if (mobileMenu.classList.contains('hidden')) {
      icon.setAttribute('data-lucide', 'menu');
    } else {
      icon.setAttribute('data-lucide', 'x');
    }
    lucide.createIcons(); // アイコンを再描画
  });

  /*   const form = document.getElementById('chatbot-form');
     const input = document.getElementById('chatbot-input');
     const messages = document.getElementById('chatbot-messages');
     form.addEventListener('submit', function(e) {
       e.preventDefault();
       const userMsg = input.value.trim();
       if (!userMsg) return;
       messages.innerHTML += `<div style="margin-bottom:8px;"><b>あなた:</b> ${userMsg}</div>`;
       input.value = '';
       // ダミー応答（API連携部分をここに実装）
       setTimeout(() => {
         messages.innerHTML += `<div style="margin-bottom:8px;"><b>Bot:</b> 申し訳ありません、現在はデモ応答のみです。</div>`;
         messages.scrollTop = messages.scrollHeight;
       }, 600);
      messages.scrollTop = messages.scrollHeight;
     });
  */
  const form = document.getElementById('chatbot-form');
  const input = document.getElementById('chatbot-input');
  const messages = document.getElementById('chatbot-messages');
  const imageInput = document.getElementById('chatbot-image');

  imageInput.addEventListener('change', function () {
    const imageFile = imageInput.files[0];
    if (imageFile) {
      const reader = new FileReader();
      reader.onload = function (e) {
        messages.innerHTML += `<div style="margin-bottom:8px;"><b>You:</b> Image Uploaded<br>
        <img src="${e.target.result}" alt="Uploaded Image" style="max-width:120px; max-height:120px; border-radius:8px; margin-top:4px;"/>`;
        messages.scrollTop = messages.scrollHeight;
      };
      reader.readAsDataURL(imageFile);
    }
  });

  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    try {
      showChatLoading(true);
      const userMsg = input.value.trim();
      const imageFile = imageInput.files[0];

      if (userMsg) {
        const userLabel = currentUser ? `${currentUser}` : "You";
        messages.innerHTML += `<div style="margin-bottom:8px;"><b>${userLabel}:</b> ${userMsg}</div>`;
        input.value = '';
        const formData = new FormData();
        formData.append('email', currentUser ? currentUser : 'unknown');
        formData.append('question', userMsg);
        const res = await fetch('http://localhost:8000/chatbot_answer', {
          method: 'POST',
          body: formData
        });
        if (!res.ok) throw new Error(`API Error: ${res.status}`);
        const html = await res.text();
        document.getElementById('chatbot-messages').innerHTML += html;
      }

      if (imageFile) {
        const formData = new FormData();
        formData.append('file', imageFile);
        const response = await fetch('http://localhost:8000/recommendwithselected', {
          method: 'POST',
          body: formData
        });
        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        const data = await response.json();
        console.log("API Response:", data);
        if (data.recommendations) {
          messages.innerHTML += `<div style="margin-bottom:8px;"><b>Bot:</b> Recommend Items:</div>`;
          messages.innerHTML += data.recommendations.html;
        } else {
          messages.innerHTML += `<div style="margin-bottom:8px;"><b>Bot:</b> Recommend Items not found.</div>`;
        }
        messages.scrollTop = messages.scrollHeight;
        imageInput.value = '';
      }
    } catch (err) {
      messages.innerHTML += `<div style="color:red; margin-bottom:8px;"><b>エラー:</b> ${err.message}</div>`;
    } finally {
      showChatLoading(false);
    }
    messages.scrollTop = messages.scrollHeight;
  });

  // ログインモーダル表示（例: ユーザーアイコン押下で表示）
  const userBtn = document.getElementById('user-btn');
  if (userBtn) {
    userBtn.addEventListener('click', () => {
      document.getElementById('login-modal').style.display = 'flex';
    });
  }

  // ログイン処理（Mock）
  document.getElementById('login-btn').addEventListener('click', async function () {
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    if (username && password) {
      currentUser = username; // ユーザー名を保持
      document.getElementById('login-modal').style.display = 'none';
      // チャット欄にログインユーザー名を表示
      const messages = document.getElementById('chatbot-messages');
      messages.innerHTML += `<div style="margin-bottom:8px; color:#374151;"> ${currentUser} logged in</div>`;

      try {
        showLoading(true);
        await Promise.all([
          fetchBestBanner(username),
          fetchRecommendItems(username)
        ]);
      } catch (err) {
        alert("エラー: " + err.message);
      } finally {
        showLoading(false);
      }
    } else {
      alert('Please enter your username and password');
    }
  });

  // jsではCurrent User = email
  async function fetchBestBanner(email) {
    const formData = new FormData();
    formData.append("email", email);

    try {
      const res = await fetch("http://localhost:8000/select_banner", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.banner_path) {
        // バナー画像を表示
        const heroBanner = document.getElementById("hero-banner");
        if (heroBanner) {
          heroBanner.style.backgroundImage = `url('${data.banner_path}')`;
        }
        // 理由を表示（任意）
        if (data.reason) {
          console.log("Selected Banner Reason:", data.reason);
        }
      } else if (data.error) {
        alert("バナー選択エラー: " + data.error);
      }
    } catch (err) {
      alert("通信エラー: " + err);
    }
  }

  async function fetchRecommendItems(email) {
    const formData = new FormData();
    formData.append("email", email);
    try {
      const res = await fetch("http://localhost:8000/recommend_items", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.items) {
        const grid = document.querySelector('.grid.grid-cols-2.md\\:grid-cols-4');
        grid.innerHTML = ""; // 既存カードを消す
        data.items.forEach(item => {
          grid.innerHTML += `
            <div class="group">
              <div class="aspect-[3/4] bg-gray-100 rounded-lg overflow-hidden mb-2">
                <img src="${item.img}" alt="${item.name}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
             </div>
             <h3 class="font-bold">${item.name}</h3>
             <p class="text-gray-600">${"$" + item.price}</p>
           </div>
          `;
        });
      } else if (data.error) {
        alert("おすすめ取得エラー: " + data.error);
      }
    } catch (err) {
      alert("通信エラー: " + err);
    }

  }



  // モーダル外クリックで閉じる
  document.getElementById('login-modal').addEventListener('click', function (e) {
    if (e.target === this) this.style.display = 'none';
  });

  // チャットボット開閉ボタン
  const chatbotContainer = document.getElementById('chatbot-container');
  const chatbotToggleBtn = document.getElementById('chatbot-toggle-btn');
  const chatbotCloseBtn = document.getElementById('chatbot-close-btn');

  chatbotToggleBtn.addEventListener('click', function () {
    chatbotContainer.style.display = 'flex';
    chatbotToggleBtn.style.display = 'none';
  });

  chatbotCloseBtn.addEventListener('click', function () {
    chatbotContainer.style.display = 'none';
    chatbotToggleBtn.style.display = 'block';
  });

});

// ローディング表示制御
function showLoading(show) {
  const overlay = document.getElementById('loading-overlay');
  overlay.style.display = show ? 'flex' : 'none';
}

function showChatLoading(show) {
  document.getElementById('chatbot-loading').style.display = show ? 'block' : 'none';
}



//検索エンジン
const query = document.getElementById("query");
const ul = document.getElementById("suggest");
let timer, ctrl;

function debounce(fn, ms) { return (...a) => { clearTimeout(timer); timer = setTimeout(() => fn(...a), ms); }; }

async function call(path, body) {
  if (ctrl) ctrl.abort();
  ctrl = new AbortController();
  try {
    const r = await fetch(`http://localhost:8000/${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: ctrl.signal
    });
    return r.json();
  } catch (err) {
    if (err.name === "AbortError") return; // AbortErrorは無視
    throw err;
  }
}

query.addEventListener("input", debounce(async e => {
  const prefix = e.target.value.trim();
  if (prefix.length < 3) {
    ul.innerHTML = "";
    ul.style.display = "none";
    return;
  }
  if (!prefix) { ul.innerHTML = ""; return; }

  // completion
  const s = await call("suggest", { prefix });
  // typeahead（切替テストしたい時はこっち）
  // const s = await call("typeahead", { prefix });

  ul.innerHTML = (s.suggestions
    ? s.suggestions.map(t => ({ name: t, id: "" })) // suggestionsが文字列のみの場合
    : s.hits.map(h => ({ name: h.productDisplayName, id: h.id }))
  ).map(t =>
    `<li onclick="document.getElementById('query').value='${t.name}'; ul.style.display='none';">
      <img src="examples/data/sample_clothes/sample_images/${t.id}.jpg" class="suggest-img";">
     <span>${t.name}</span>
    </li>`
  ).join('');
  ul.style.display = ul.innerHTML ? "block" : "none";
}, 120));

async function search({ q, filters = {}, sort = "score", page = 1, per_page = 24 }) {
  const r = await fetch("http://localhost:8000/search", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q, filters, sort, page, per_page })
  });
  return r.json();
}

function showSuggest(items) {
  if (items.length === 0) {
    suggest.style.display = "none";
    return;
  }
  suggest.innerHTML = items.map(item =>
    `<li style="padding:8px 12px; cursor:pointer; border-bottom:1px solid #f3f4f6;" 
      onmouseover="this.style.background='#f3f4f6'" 
      onmouseout="this.style.background=''" 
      onclick="document.getElementById('query').value='${item}'; suggest.style.display='none';"
    >${item}</li>`
  ).join('');
  suggest.style.display = "block";
}

// 検索実行関数
// 検索実行＆描画関数
async function searchAndRender({ sort = "score", page = 1, per_page = 24 } = {}) {
  const queryInput = document.getElementById("query");
  const q = queryInput.value.trim();
  const main = document.querySelector("main");

  // ローディング表示
  main.innerHTML = `<div style="text-align:center; color:#374151; padding:48px 0;">Searching...</div>`;

  try {
    // 検索API呼び出し
    const res = await search({ q, sort, page, per_page });

    // OpenSearch形式のパース
    const items = (res.hits && res.hits.hits)
      ? res.hits.hits.map(hit => hit._source)
      : [];

    if (!items.length) {
      main.innerHTML = `<div style="text-align:center; color:#374151; padding:48px 0;">No results found.</div>`;
      return;
    }

    // 並び替えUI
    const sortOptions = [
      { value: "score", label: "Relevance" },
      { value: "price_asc", label: "Price: Low to High" },
      { value: "price_desc", label: "Price: High to Low" },
    ];
    const sortSelect = `
      <div class="text-right mb-4">
        <label for="sort-select" style="margin-right:8px;">Sort by:</label>
        <select id="sort-select" style="padding:4px 8px; border-radius:4px; border:1px solid #e5e7eb;">
          ${sortOptions.map(opt => `<option value="${opt.value}"${sort === opt.value ? " selected" : ""}>${opt.label}</option>`).join("")}
        </select>
      </div>
    `;

    // 商品リストHTML
    const itemsHtml = `
      <section class="py-16">
        <div class="container mx-auto px-4">
          <h2 class="text-3xl font-bold text-center mb-8">Search Results</h2>
          ${sortSelect}
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-8">
            ${items.map(item => `
              <div class="group">
                <div class="aspect-[3/4] bg-gray-100 rounded-lg overflow-hidden mb-2">
                  <img
                    src="examples/data/sample_clothes/sample_images/${item.id}.jpg"
                    alt="${item.productDisplayName}"
                    class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                </div>
                <h3 class="font-bold">${item.productDisplayName}</h3>
                <p class="text-gray-600">$${item.price}</p>
              </div>
            `).join("")}
          </div>
        </div>
      </section>
    `;

    main.innerHTML = itemsHtml;

    // 並び替えイベント
    document.getElementById("sort-select").onchange = function () {
      searchAndRender({ sort: this.value, page: 1, per_page });
    };
  } catch (err) {
    main.innerHTML = `<div style="color:red; text-align:center; padding:48px 0;">Search error: ${err.message}</div>`;
  }
}

// 検索ボタン・Enterキーで検索実行
document.getElementById("search-btn").addEventListener("click", () => searchAndRender());
document.getElementById("query").addEventListener("keydown", function (e) {
  if (e.key === "Enter") searchAndRender();
});

document.getElementById("logo-link").addEventListener("click", function (e) {
  e.preventDefault();
  window.scrollTo(0, 0); // まずトップにスクロール
  window.location.reload();
});