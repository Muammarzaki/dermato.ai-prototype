package data

import (
	"context"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type Chronic struct {
	ID        uuid.UUID `gorm:"type:uuid;primary_key;default:gen_random_uuid()" json:"id"`
	Body      string    `gorm:"type:json" json:"body"`
	Status    string    `gorm:"type:varchar(10);check:status IN ('success','fail')" json:"status"`
	CreatedAt time.Time `gorm:"type:timestamp;not null" json:"created_at"`
}

type ChronicRepository struct {
	db *gorm.DB
}

func NewChronicRepository(db *gorm.DB) *ChronicRepository {
	return &ChronicRepository{
		db: db,
	}
}

func (r *ChronicRepository) Create(ctx context.Context, chronic *Chronic) error {
	return r.db.WithContext(ctx).Create(chronic).Error
}
