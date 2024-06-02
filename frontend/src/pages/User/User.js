import React, { useState, useEffect } from "react";
import Article from "../../components/Article/Article";
import styles from "./User.module.css";
import axios from "axios";

function User() {
  const [articles, setArticles] = useState([]);
  const [bookmarks, setBookmarks] = useState([]);
  const [error, setError] = useState(null);

  // 사용자 저장된 기사 가져오기 함수
  useEffect(() => {
    async function fetchArticles() {
      try {
        const token = localStorage.getItem("token");
        if (!token) {
          throw new Error("No token found");
        }

        const response = await axios.get(
          "http://localhost:8000/api/article/scrape/",
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
            withCredentials: true, // 쿠키를 포함한 요청
          }
        );

        const scrapedArticles = response.data.map((scrape) => scrape.article);
        setArticles(scrapedArticles);
      } catch (error) {
        console.error("Error fetching articles:", error);
      }
    }

    async function fetchBookmarks() {
      try {
        const token = localStorage.getItem("token");
        if (!token) {
          throw new Error("No token found");
        }

        const response = await axios.get("http://localhost:8000/accounts/bookmarks/list/", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        setBookmarks(response.data);
      } catch (error) {
        setError("Failed to fetch bookmarks.");
        console.error("Error fetching bookmarks:", error);
      }
    }

    fetchArticles();
    fetchBookmarks();
  }, []);

  const handleBookmarkToggle = async (bookmarkId, url, title, summary) => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("No token found");
      }

      const response = await axios.post(
        "http://localhost:8000/accounts/bookmarks/",
        {
          url: url,
          title: title,
          summary: summary,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.data.status === "bookmark removed") {
        alert("Bookmark removed successfully");
        setBookmarks(bookmarks.filter((bookmark) => bookmark.id !== bookmarkId));
      } else {
        alert("Bookmark added successfully");
      }
    } catch (error) {
      setError("Failed to toggle bookmark.");
      console.error("Error toggling bookmark:", error);
    }
  };

  if (error) {
    return <div>{error}</div>;
  }

  return (
    <div className={styles.container}>
      <div className={styles.content_wrapper}>
        <p className={styles.title}>마이페이지</p>
        <hr />
        <div className={styles.article_container}>
          {articles.map((article) => (
            <Article key={article.id} {...article} />
          ))}
        </div>
        <hr />
        <div className={styles.bookmark_container}>
          <p className={styles.subtitle}>북마크된 기사</p>
          {bookmarks.map((bookmark) => (
            <div key={bookmark.id} className={styles.bookmark}>
              <a href={bookmark.url} target="_blank" rel="noopener noreferrer">
                {bookmark.title}
              </a>
              <p>{bookmark.summary}</p>
              <p>{new Date(bookmark.created_at).toLocaleString()}</p>
              <button
                onClick={() =>
                  handleBookmarkToggle(
                    bookmark.id,
                    bookmark.url,
                    bookmark.title,
                    bookmark.summary
                  )
                }
              >
                Remove Bookmark
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default User;