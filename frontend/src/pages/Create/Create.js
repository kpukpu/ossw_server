import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import styles from "./Create.module.css";
import title from "../../assets/images/title.png";
import send from "../../assets/images/send.png";

function Create() {
  const navigate = useNavigate();
  const [url, setUrl] = useState("");

  // URL 입력 변경 시 호출되는 함수
  const handleInputChange = (event) => {
    setUrl(event.target.value);
  };

  // URL 전송 로직
  const handleSendClick = () => {
    navigate(`/result?url=${encodeURIComponent(url)}`);
  };

  return (
    <div className={styles.container}>
      <div className={styles.imageContainer}>
        <img src={title} alt="title" />
        <p className={styles.textOverImage}>요약된 기사와 분석을 제공합니다.</p>
      </div>

      <div className={styles.content_wrapper}>
        <p className={styles.title}>기사 링크 입력</p>
        <div className={styles.input_container}>
          <input
            className={styles.input}
            value={url}
            onChange={handleInputChange}
            placeholder="기사 URL 입력"
          />
          <img
            src={send}
            alt="send"
            onClick={handleSendClick}
            style={{ cursor: "pointer" }}
          />
        </div>
      </div>
    </div>
  );
}

export default Create;
